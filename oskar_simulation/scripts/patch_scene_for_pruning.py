#!/usr/bin/env python3
"""Prepare the Isaac Sim orchard scene for the real-time pruning stack.

Three changes, all idempotent:

1. Reparent ZED_X under wrist_3_link and make it passive. The ZED X asset
   applies PhysicsRigidBodyAPI at its root, so the camera is a rigid body held
   to the wrist only by fixed_camera_joint. A sensor needs no dynamics, and a
   dynamic camera is one solver failure away from drifting off its mount, so
   the body and its collider are disabled and the camera rides the link
   through the hierarchy instead. The joint goes with them.

2. Add a GroundTruth graph publishing TF for the whole articulation from
   physics: world -> base_link and every link below it, the pose the pruning
   nodes localize with.

3. Add a WristCamera graph publishing the ZED's left and right eyes plus a
   depth image. This bypasses the ZED SDK path (the sl.sensor.camera extension
   and pyzed), which the perception node does not need and which pins NumPy 2.

Optionally it also labels every flower prim, which lets the renderer publish
instance segmentation. That gives perfect masks on the real rendered image, so
the geometry and timing can be brought up without the detector in the loop, and
the detector can afterwards be measured against a known ceiling.

It also vendors the ZED X body, which the scene otherwise references from
NVIDIA's S3 bucket. Without it the camera prims compose as empty placeholders
that still look valid, so anything reading the camera silently gets the mount
point's transform instead. Vendoring keeps the reference relative, as every
other asset in this scene already is.

Run with Isaac's interpreter, which ships pxr:
  ~/isaacsim_6_0_1/kit/python/bin/python3 patch_scene_for_pruning.py <scene.usd> [--output OUT | --in-place]
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdSemantics

ROBOT = "/tipard_ur20_combined"
WRIST = f"{ROBOT}/wrist_3_link"
BASE_LINK = f"{ROBOT}/base_link"
ZED = f"{ROBOT}/ZED_X"
ZED_MOUNTED = f"{WRIST}/ZED_X"

# Camera prims inside the referenced ZED X asset (its default prim is /Root).
ZED_LEFT = "base_link/ZED_X/CameraLeft"
ZED_RIGHT = "base_link/ZED_X/CameraRight"

ORCHARD = "/tipard_ur20_combined/two_by_two_orchard"
FLOWER_PATTERN = re.compile(r"^Flower_C(\d+)_(K|S)_(\d+)$")

ZED_GRAPH = "/World/ZED_Camera"
STATIC_CAMERA_GRAPHS = ("/World/static_camera_1", "/World/static_camera_2")
ARM_GRAPH = "/World/Arm_gripper"

GT_GRAPH = "/World/GroundTruth"
CAM_GRAPH = "/World/WristCamera"

DEFAULT_ZED_ASSET = Path.home() / "isaac_sim_assets/Sensors/Stereolabs/ZED_X/ZED_X.usd"
VENDORED_ZED = "sensors/ZED_X/ZED_X.usd"


def node(stage, graph, name, node_type, type_version=1):
    prim = stage.DefinePrim(f"{graph}/{name}", "OmniGraphNode")
    prim.CreateAttribute("node:type", Sdf.ValueTypeNames.Token).Set(node_type)
    prim.CreateAttribute("node:typeVersion", Sdf.ValueTypeNames.Int).Set(type_version)
    return prim


def connect(prim, input_name, type_name, source):
    prim.CreateAttribute(f"inputs:{input_name}", type_name).SetConnections([Sdf.Path(source)])


def define_graph(stage, path):
    prim = stage.DefinePrim(path, "OmniGraph")
    prim.CreateAttribute("evaluationMode", Sdf.ValueTypeNames.Token).Set("Automatic")
    prim.CreateAttribute("evaluator:type", Sdf.ValueTypeNames.Token).Set("execution")
    prim.CreateAttribute("fabricCacheBacking", Sdf.ValueTypeNames.Token).Set("StageWithoutHistory")
    prim.CreateAttribute("pipelineStage", Sdf.ValueTypeNames.Token).Set("pipelineStageSimulation")
    prim.CreateAttribute("fileFormatVersion", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(1, 9))
    return prim


def vendor_zed_asset(stage, scene_path, source):
    """Copy the ZED X body next to the scene and point the reference at it."""
    scene_dir = Path(scene_path).resolve().parent
    dest = scene_dir / VENDORED_ZED

    zed = stage.GetPrimAtPath(ZED) or stage.GetPrimAtPath(ZED_MOUNTED)
    if not zed:
        print(f"  ERROR: no ZED_X prim at {ZED} or {ZED_MOUNTED}")
        return False

    refs = zed.GetMetadata("references")
    current = [r.assetPath for r in (refs.prependedItems + refs.appendedItems)] if refs else []
    if current == [VENDORED_ZED] and dest.exists():
        print(f"  ZED X body already vendored at {VENDORED_ZED}")
        return True

    if not dest.exists():
        if not source.exists():
            print(f"  ERROR: no local ZED X asset at {source}; pass --zed-asset")
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        print(f"  vendored ZED X body: {source} -> {VENDORED_ZED}")

    zed.GetReferences().ClearReferences()
    zed.GetReferences().AddReference(VENDORED_ZED)
    print(f"  ZED_X now references {VENDORED_ZED} instead of NVIDIA's S3 bucket")
    return True


def clear_stale_render_products(stage):
    """Drop render products left in the scene by earlier sessions.

    These are runtime state that only reaches the file when a scene is saved
    while playing. On the next run the renderer finds them without their
    settings and reports "Missing cached render settings for render product".
    """
    stale = [p.GetPath() for p in stage.Traverse()
             if p.GetTypeName() == "RenderProduct" and p.GetName().startswith("Replicator")]
    if not stale:
        print("  no stale render products")
        return True
    edit = Sdf.BatchNamespaceEdit()
    for path in stale:
        edit.Add(Sdf.NamespaceEdit.Remove(path))
    if not stage.GetRootLayer().Apply(edit):
        print(f"  WARNING: could not remove {len(stale)} stale render products")
        return True
    print(f"  removed {len(stale)} stale render products left by earlier sessions")
    return True


def label_flowers(stage):
    """Tag each flower prim so the renderer can segment them.

    The label goes on the flower group, not its petal and stamen meshes, so one
    flower is one instance and the segmentation ids line up with the ground
    truth exported by export_orchard_gt.py.
    """
    orchard = stage.GetPrimAtPath(ORCHARD)
    if not orchard:
        print(f"  ERROR: no orchard at {ORCHARD}")
        return False
    labelled = 0
    for tree in orchard.GetChildren():
        if not tree.GetName().startswith("AppleTree"):
            continue
        for child in tree.GetChildren():
            if not FLOWER_PATTERN.match(child.GetName()):
                continue
            UsdSemantics.LabelsAPI.Apply(child, "class").CreateLabelsAttr(["flower"])
            labelled += 1
    print(f"  labelled {labelled} flower prims as class 'flower'")
    return True


def mount_camera_on_wrist(stage):
    """Move ZED_X under wrist_3_link, keeping its world pose."""
    if stage.GetPrimAtPath(ZED_MOUNTED):
        print("  camera already mounted on wrist_3_link")
        return True
    zed = stage.GetPrimAtPath(ZED)
    wrist = stage.GetPrimAtPath(WRIST)
    if not zed or not wrist:
        print(f"  ERROR: need both {ZED} and {WRIST}")
        return False

    world = UsdGeom.Xformable(zed).ComputeLocalToWorldTransform(0)
    wrist_world = UsdGeom.Xformable(wrist).ComputeLocalToWorldTransform(0)
    local = world * wrist_world.GetInverse()

    joint = f"{ZED}/fixed_camera_joint"
    edit = Sdf.BatchNamespaceEdit()
    if stage.GetPrimAtPath(joint):
        edit.Add(Sdf.NamespaceEdit.Remove(joint))
    edit.Add(Sdf.NamespaceEdit(Sdf.Path(ZED), Sdf.Path(ZED_MOUNTED)))
    if not stage.GetRootLayer().Apply(edit):
        print("  ERROR: namespace edit rejected -- ZED_X is probably not defined in the root layer")
        return False

    prim = stage.GetPrimAtPath(ZED_MOUNTED)
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    # Clearing the order leaves the old ops behind as unused attributes, and
    # Isaac still writes to them; authoring the same translate/orient/scale
    # stack the prim already had keeps types and precision consistent.
    for attribute in list(prim.GetAttributes()):
        if attribute.GetName().startswith("xformOp:"):
            prim.RemoveProperty(attribute.GetName())

    t = local.ExtractTranslation()
    q = local.ExtractRotationQuat()
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(t))
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q))
    xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))

    disabled = 0
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        disabled += 1
    for descendant in Usd.PrimRange(prim):
        if descendant.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(descendant).CreateCollisionEnabledAttr(False)
            disabled += 1

    print(f"  mounted ZED_X on wrist_3_link at local ({t[0]:.5f}, {t[1]:.5f}, {t[2]:.5f}); "
          f"removed fixed_camera_joint and disabled {disabled} physics bodies or colliders "
          "so the camera rides the link instead of being simulated")
    return True


def retarget_arm_controller(stage):
    """Point the arm's graph at the articulation root.

    IsaacArticulationController needs the prim carrying
    PhysicsArticulationRootAPI, which is base_link. Aimed at ur20_base_link it
    finds no articulation and every joint command is dropped.
    """
    root = Sdf.Path(BASE_LINK)
    fixed = []
    for name in ("articulation_controller", "ros2_publish_joint_state"):
        prim = stage.GetPrimAtPath(f"{ARM_GRAPH}/{name}")
        if not prim:
            continue
        rel = prim.GetRelationship("inputs:targetPrim")
        if rel and rel.GetTargets() != [root]:
            rel.SetTargets([root])
            fixed.append(name)
    if fixed:
        print(f"  retargeted {', '.join(fixed)} from ur20_base_link to the "
              "articulation root, base_link")
    return True


def remove_static_cameras(stage):
    """Drop the base-mounted stereo pair.

    Each is its own render product, so the scene renders three cameras when the
    pruning stack reads one, and rendering is what caps the frame rate. They
    belong to the Phase 1 drive-by rig and aim down the alley, not at a tree.
    """
    present = [path for path in STATIC_CAMERA_GRAPHS if stage.GetPrimAtPath(path)]
    if not present:
        print("  no static camera graphs to remove")
        return True
    edit = Sdf.BatchNamespaceEdit()
    for path in present:
        edit.Add(Sdf.NamespaceEdit.Remove(Sdf.Path(path)))
    if not stage.GetRootLayer().Apply(edit):
        print("  WARNING: could not remove the static camera graphs")
        return True
    print(f"  removed {len(present)} static camera graphs, "
          "one render product each, unused by this stack")
    return True


def remove_zed_sdk_graph(stage):
    """Drop the Stereolabs streaming graph.

    It addresses the camera at its old path and needs the ZED SDK, which this
    stack deliberately bypasses; left in place it logs an error every frame.
    """
    if not stage.GetPrimAtPath(ZED_GRAPH):
        print("  no ZED SDK graph to remove")
        return True
    edit = Sdf.BatchNamespaceEdit()
    edit.Add(Sdf.NamespaceEdit.Remove(Sdf.Path(ZED_GRAPH)))
    if not stage.GetRootLayer().Apply(edit):
        print(f"  WARNING: could not remove {ZED_GRAPH}")
        return True
    print(f"  removed {ZED_GRAPH}, the ZED SDK streaming graph this stack does not use")
    return True


def add_ground_truth_graph(stage):
    """Publish TF for the articulation straight from physics."""
    define_graph(stage, GT_GRAPH)
    ctx = node(stage, GT_GRAPH, "ros2_context", "isaacsim.ros2.bridge.ROS2Context", 2)
    tick = node(stage, GT_GRAPH, "on_playback_tick", "omni.graph.action.OnPlaybackTick", 2)
    clock = node(stage, GT_GRAPH, "read_sim_time", "isaacsim.core.nodes.IsaacReadSimulationTime", 1)

    tf = node(stage, GT_GRAPH, "publish_tf", "isaacsim.ros2.bridge.ROS2PublishTransformTree", 1)
    connect(tf, "context", Sdf.ValueTypeNames.UInt64, f"{ctx.GetPath()}.outputs:context")
    connect(tf, "execIn", Sdf.ValueTypeNames.UInt, f"{tick.GetPath()}.outputs:tick")
    connect(tf, "timeStamp", Sdf.ValueTypeNames.Double, f"{clock.GetPath()}.outputs:simulationTime")
    tf.CreateAttribute("inputs:topicName", Sdf.ValueTypeNames.String).Set("tf")
    # No parentPrim: the publisher then names the parent frame "world", which the
    # launch file ties to "map" with a static identity transform.
    tf.CreateRelationship("inputs:targetPrims").SetTargets([Sdf.Path(BASE_LINK)])
    print(f"  {GT_GRAPH}: TF for the articulation rooted at {BASE_LINK}, parent frame 'world'")


def add_wrist_camera_graph(stage, width, height, labels, both_eyes, rgb):
    """Publish the wrist camera and a depth image.

    Each eye is a separate render product, so a full extra render pass per
    frame. The right eye only feeds a stereo depth path that nothing uses yet,
    so it is off by default: the frame rate is what bounds how fast the arm may
    sweep, and buys more than the unused stream.
    """
    define_graph(stage, CAM_GRAPH)
    ctx = node(stage, CAM_GRAPH, "ros2_context", "isaacsim.ros2.bridge.ROS2Context", 2)
    tick = node(stage, CAM_GRAPH, "on_playback_tick", "omni.graph.action.OnPlaybackTick", 2)

    # A render product must be created once, not once per tick. Driving it
    # straight from the tick allocates a new one every frame until the renderer
    # dies, which is what the scene's own cameras avoid with this same node.
    run_once = node(stage, CAM_GRAPH, "run_once",
                    "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame", 1)
    connect(run_once, "execIn", Sdf.ValueTypeNames.UInt, f"{tick.GetPath()}.outputs:tick")

    eyes = [("left", ZED_LEFT)] + ([("right", ZED_RIGHT)] if both_eyes else [])
    for eye, rel_path in eyes:
        cam_path = f"{ZED_MOUNTED}/{rel_path}"
        rp = node(stage, CAM_GRAPH, f"render_product_{eye}",
                  "isaacsim.core.nodes.IsaacCreateRenderProduct", 1)
        connect(rp, "execIn", Sdf.ValueTypeNames.UInt, f"{run_once.GetPath()}.outputs:step")
        rp.CreateAttribute("inputs:width", Sdf.ValueTypeNames.UInt).Set(width)
        rp.CreateAttribute("inputs:height", Sdf.ValueTypeNames.UInt).Set(height)
        rp.CreateRelationship("inputs:cameraPrim").SetTargets([Sdf.Path(cam_path)])

        # rgb and camera_info on both eyes; depth only on the left, which is the
        # frame the perception node reasons in.
        products = []
        if rgb:
            products.append(("rgb", "rgb", f"/wrist_cam/{eye}/image_raw"))
        if eye == "left":
            products.append(("depth", "depth", "/wrist_cam/left/depth"))
            if labels:
                products.append(("instances", "instance_segmentation",
                                 "/wrist_cam/left/instance_segmentation"))

        for suffix, kind, topic in products:
            pub = node(stage, CAM_GRAPH, f"publish_{eye}_{suffix}",
                       "isaacsim.ros2.bridge.ROS2CameraHelper", 2)
            connect(pub, "context", Sdf.ValueTypeNames.UInt64, f"{ctx.GetPath()}.outputs:context")
            connect(pub, "execIn", Sdf.ValueTypeNames.UInt, f"{rp.GetPath()}.outputs:execOut")
            connect(pub, "renderProductPath", Sdf.ValueTypeNames.Token,
                    f"{rp.GetPath()}.outputs:renderProductPath")
            pub.CreateAttribute("inputs:type", Sdf.ValueTypeNames.Token).Set(kind)
            pub.CreateAttribute("inputs:topicName", Sdf.ValueTypeNames.String).Set(topic)
            pub.CreateAttribute("inputs:frameId", Sdf.ValueTypeNames.String).Set(f"wrist_cam_{eye}")
            pub.CreateAttribute("inputs:resetSimulationTimeOnStop", Sdf.ValueTypeNames.Bool).Set(True)
            if kind == "instance_segmentation":
                # The id image alone is anonymous; the labels topic is what maps
                # each id back to the flower prim it came from.
                pub.CreateAttribute("inputs:enableSemanticLabels",
                                    Sdf.ValueTypeNames.Bool).Set(True)
                pub.CreateAttribute("inputs:semanticLabelsTopicName",
                                    Sdf.ValueTypeNames.String).Set(
                                        "/wrist_cam/left/instance_labels")

        info = node(stage, CAM_GRAPH, f"publish_{eye}_camera_info",
                    "isaacsim.ros2.bridge.ROS2CameraInfoHelper", 1)
        connect(info, "context", Sdf.ValueTypeNames.UInt64, f"{ctx.GetPath()}.outputs:context")
        connect(info, "execIn", Sdf.ValueTypeNames.UInt, f"{rp.GetPath()}.outputs:execOut")
        connect(info, "renderProductPath", Sdf.ValueTypeNames.Token,
                f"{rp.GetPath()}.outputs:renderProductPath")
        info.CreateAttribute("inputs:topicName", Sdf.ValueTypeNames.String).Set(
            f"/wrist_cam/{eye}/camera_info")
        info.CreateAttribute("inputs:frameId", Sdf.ValueTypeNames.String).Set(f"wrist_cam_{eye}")
        info.CreateAttribute("inputs:resetSimulationTimeOnStop", Sdf.ValueTypeNames.Bool).Set(True)

    extra = " + instance_segmentation and instance_labels" if labels else ""
    which = "{left,right}" if both_eyes else "left"
    colour = f"/wrist_cam/{which}/image_raw + " if rgb else "no colour stream, "
    print(f"  {CAM_GRAPH}: {colour}camera_info, "
          f"/wrist_cam/left/depth{extra}, at {width}x{height} "
          f"({len(eyes)} render product{'s' if len(eyes) > 1 else ''})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--output", help="write the patched scene here")
    ap.add_argument("--in-place", action="store_true", help="overwrite the input scene")
    # ZED X renders 1920x1200 per eye; the default trades resolution for frame rate.
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--zed-asset", type=Path, default=DEFAULT_ZED_ASSET,
                    help="local copy of the ZED X body to vendor into the scene")
    ap.add_argument("--no-flower-labels", action="store_true",
                    help="skip semantic labelling, and the instance segmentation stream")
    ap.add_argument("--no-rgb", action="store_true",
                    help="skip the colour stream, which only the maskrcnn detector needs; "
                         "saves 2.8 MB a frame at 1280x720")
    ap.add_argument("--keep-zed-graph", action="store_true",
                    help="keep the Stereolabs streaming graph")
    ap.add_argument("--keep-static-cameras", action="store_true",
                    help="keep the base-mounted stereo pair and its two render products")
    ap.add_argument("--both-eyes", action="store_true",
                    help="also render the right eye, for a stereo depth path")
    args = ap.parse_args()

    if bool(args.output) == args.in_place:
        sys.exit("pass exactly one of --output or --in-place")

    stage = Usd.Stage.Open(args.scene)
    print(f"patching {args.scene}")
    target = args.scene if args.in_place else args.output
    if not vendor_zed_asset(stage, target, args.zed_asset):
        sys.exit(1)
    if not mount_camera_on_wrist(stage):
        sys.exit(1)
    clear_stale_render_products(stage)
    retarget_arm_controller(stage)
    if not args.keep_zed_graph:
        remove_zed_sdk_graph(stage)
    if not args.keep_static_cameras:
        remove_static_cameras(stage)
    labels = not args.no_flower_labels
    if labels and not label_flowers(stage):
        sys.exit(1)
    add_ground_truth_graph(stage)
    add_wrist_camera_graph(stage, args.width, args.height, labels, args.both_eyes,
                           not args.no_rgb)

    if args.in_place:
        stage.Save()
        print(f"saved in place: {args.scene}")
    else:
        stage.GetRootLayer().Export(args.output)
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
