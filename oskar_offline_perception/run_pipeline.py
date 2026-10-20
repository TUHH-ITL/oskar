#!/usr/bin/env python3
"""Main coordinator script to run the modular offline apple flower mapping pipeline in-memory."""

import json
import os
import sys
import time

# Scrub ROS 2 statistics package module shadowing
sys.path = [p for p in sys.path if "statistics" not in p]

import queue
import threading
import pickle
import gzip

import config
import cv2
import numpy as np
import utils
from step1_segmentation import load_segmentation_model
if config.SEG_TILED_INFERENCE:
    from step1_segmentation_tiled import segment_frame
else:
    from step1_segmentation import segment_frame
from step2_disparity import load_disparity_model, disparity_frame, disparity_frame_full
from step3_depth_fusion import depth_fusion_frame
from step4_backprojection import get_base_to_camera_transform, interpolate_pose
from step5_multiview_fusion import process_multiview_fusion
from step5_tracking_fusion import process_tracking_fusion
from step5_voxel_fusion import process_voxel_fusion
from step6_bio_sanity import process_bio_sanity
from step7_tree_assignment import process_tree_assignment, load_tree_map
from step8_thinning_decision import process_thinning_decision


def prefetch_worker(image_pairs, rect_maps, prefetch_queue):
    """Background loader thread that reads and rectifies images on the CPU."""
    (map_lx, map_ly), (map_rx, map_ry) = rect_maps
    for left_path, right_path in image_pairs:
        left_cv = cv2.imread(left_path)
        right_cv = cv2.imread(right_path)
        if left_cv is not None:
            left_cv = utils.normalize_calibration_orientation(left_cv, utils.load_calibration_yaml(config.LEFT_CALIB_FILE))
        if right_cv is not None:
            right_cv = utils.normalize_calibration_orientation(right_cv, utils.load_calibration_yaml(config.RIGHT_CALIB_FILE))
        if left_cv is not None and right_cv is not None:
            left_rect = cv2.remap(left_cv, map_lx, map_ly, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_cv, map_rx, map_ry, cv2.INTER_LINEAR)
            prefetch_queue.put((left_path, right_path, left_rect, right_rect))
        else:
            prefetch_queue.put((left_path, right_path, None, None))


def run_offline_pipeline(max_frames=None):
    """Executes the entire perception and decision mapping pipeline in-memory."""
    start_time = time.time()

    # 1. Resolve image list (Front and Back passes)
    dataset_base = os.path.dirname(config.LEFT_IMAGES_DIR)
    back_left_dir = os.path.join(dataset_base, "Basler_Left_Elstar_Back")
    back_right_dir = os.path.join(dataset_base, "Basler_Right_Elstar_Back")

    left_files = []
    right_files = []

    # Front pass left/right pairing
    if os.path.exists(config.LEFT_IMAGES_DIR) and os.path.exists(config.RIGHT_IMAGES_DIR):
        fl_files = [os.path.join(config.LEFT_IMAGES_DIR, f) for f in os.listdir(config.LEFT_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        fr_files = [os.path.join(config.RIGHT_IMAGES_DIR, f) for f in os.listdir(config.RIGHT_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        fl_dict = {os.path.basename(f).split("_")[0]: f for f in fl_files}
        fr_dict = {os.path.basename(f).split("_")[0]: f for f in fr_files}
        common_front = sorted(set(fl_dict.keys()) & set(fr_dict.keys()))
        for cid in common_front:
            left_files.append(fl_dict[cid])
            right_files.append(fr_dict[cid])
        n_front = len(common_front)
    else:
        n_front = 0

    # Back pass left/right pairing
    if os.path.exists(back_left_dir) and os.path.exists(back_right_dir):
        bl_files = [os.path.join(back_left_dir, f) for f in os.listdir(back_left_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        br_files = [os.path.join(back_right_dir, f) for f in os.listdir(back_right_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        bl_dict = {os.path.basename(f).split("_")[0]: f for f in bl_files}
        br_dict = {os.path.basename(f).split("_")[0]: f for f in br_files}
        common_back = sorted(set(bl_dict.keys()) & set(br_dict.keys()))
        # The two back-pass folders hold the cameras exchanged: fed as named, back trunks
        # measure 1.53x their surveyed distance; exchanged, 0.73x, matching the front pass's
        # 0.87x on the same 16 hand-labelled trunks.
        for cid in common_back:
            left_files.append(br_dict[cid])
            right_files.append(bl_dict[cid])
        n_back = len(common_back)
    else:
        n_back = 0

    print(f"Loaded {len(left_files)} total image pairs ({n_front} Front pass pairs + {n_back} Back pass pairs).")

    if max_frames is not None:
        left_files = left_files[:max_frames]
        right_files = right_files[:max_frames]

    print(
        f"Starting offline pipeline. Found {len(left_files)} image pairs to process.",
    )

    # 2. Build rectification maps once
    print("Building camera rectification maps...")
    rect_maps_l, rect_maps_r = utils.build_rectification_maps(
        config.LEFT_CALIB_FILE,
        config.RIGHT_CALIB_FILE,
    )

    # 3. Load Models once
    print("Loading deep learning models...")
    seg_predictor = load_segmentation_model(
        config.SEGMENTATION_MODEL,
        config.SEG_CONFIDENCE_THRESHOLD,
        device="cuda",
    )
    disparity_model = load_disparity_model()

    # Calibration parameters for depth estimation and fusion
    lc = utils.load_calibration_yaml(config.LEFT_CALIB_FILE)
    rc = utils.load_calibration_yaml(config.RIGHT_CALIB_FILE)
    K = np.array(lc["cameraMatrix"], dtype=np.float64).reshape(3, 3)

    P_l = np.array(lc["projectionMatrix"], dtype=np.float64)
    P_r = np.array(rc["projectionMatrix"], dtype=np.float64)
    fx = P_l[0, 0]
    p3 = P_r[0, 3]
    p0 = P_r[0, 0]
    baseline = (-p3 / p0) if (p0 != 0 and p3 != 0) else config.DISP_BASELINE_M

    erosion_radius = config.DEPTH_FUSION_EROSION_RADIUS
    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * erosion_radius + 1, 2 * erosion_radius + 1),
    )

    # Backprojection configuration
    camera_side = getattr(config, "CAMERA_SIDE", "right")
    camera_xyz = getattr(config, "CAMERA_XYZ", [0.1, 0.0, 1.0])
    R_base_to_cam, t_base_to_cam = get_base_to_camera_transform(
        camera_side,
        camera_xyz,
    )

    # Load trajectory once
    traj_data = np.load(config.POSE_TRAJECTORY_FILE)
    t_arr = traj_data["t"]
    E_arr = traj_data["E"]
    N_arr = traj_data["N"]
    U_arr = traj_data["U"]
    yaw_arr = traj_data["yaw"]
    cos_arr = np.cos(yaw_arr)
    sin_arr = np.sin(yaw_arr)

    # Setup prefetch queue and start background loader thread
    image_pairs = list(zip(left_files, right_files))
    prefetch_queue = queue.Queue(maxsize=4)

    loader_thread = threading.Thread(
        target=prefetch_worker,
        args=(image_pairs, (rect_maps_l, rect_maps_r), prefetch_queue),
    )
    loader_thread.daemon = True
    loader_thread.start()

    # Time tracking variables
    t_seg, t_disp, t_fuse, t_backproj = 0.0, 0.0, 0.0, 0.0
    world_observations = []
    processed_count = 0

    # Cache setup
    cache_dir = os.path.join(config.OUTPUT_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    print("\n--- Processing Frames (Steps 1 to 4 Pipelined) ---")
    try:
        for idx in range(len(image_pairs)):
            # Retrieve pre-fetched rectified frames
            left_path, right_path, left_rect, right_rect = prefetch_queue.get()
            if left_rect is None or right_rect is None:
                print(
                    f"[{idx + 1}/{len(image_pairs)}] Error reading image pair: {left_path}",
                )
                continue

            frame_basename = os.path.splitext(os.path.basename(left_path))[0]
            cache_path = os.path.join(cache_dir, f"{frame_basename}.pkl.gz")

            print(
                f"[{idx + 1}/{len(image_pairs)}] Processing frame {os.path.basename(left_path)}...",
            )

            # Check cache first
            if os.path.exists(cache_path):
                try:
                    with gzip.open(cache_path, "rb") as cf:
                        cache_data = pickle.load(cf)
                    flowers = cache_data["flowers"]
                    depth = cache_data["depth"]
                    cached = True
                except Exception as e:
                    print(f"  [Cache] Failed to load cache ({e}), running models instead.")
                    cached = False
            else:
                cached = False

            if not cached:
                # 4. Step 1: Instance Segmentation
                t0 = time.time()
                flowers = segment_frame(left_rect, seg_predictor)
                t_seg += time.time() - t0

                # 5. Step 2: Disparity Estimation
                t0 = time.time()
                if getattr(config, "DISP_FULL_FRAME", False):
                    depth = disparity_frame_full(
                        left_rect,
                        right_rect,
                        disparity_model,
                        fx,
                        baseline,
                    )
                else:
                    depth = disparity_frame(
                        left_rect,
                        right_rect,
                        flowers,
                        disparity_model,
                        fx,
                        baseline,
                    )
                t_disp += time.time() - t0

                # Save cache
                try:
                    with gzip.open(cache_path, "wb") as cf:
                        pickle.dump({"flowers": flowers, "depth": depth}, cf, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as e:
                    print(f"  [Cache] Failed to save cache ({e})")

            # 6. Step 3: Depth and Mask Fusion
            t0 = time.time()
            frame_detections = depth_fusion_frame(
                depth,
                flowers,
                K,
                erosion_kernel,
            )
            t_fuse += time.time() - t0

            # 7. Step 4: Backprojection to ENU World Frame
            t0 = time.time()
            timestamp = utils.parse_frame_timestamp(left_path)
            pose = interpolate_pose(
                timestamp,
                t_arr,
                E_arr,
                N_arr,
                U_arr,
                cos_arr,
                sin_arr,
            )
            e, n, u, yaw = pose["e"], pose["n"], pose["u"], pose["yaw"]

            cos_y = np.cos(yaw)
            sin_y = np.sin(yaw)
            R_map_to_base = np.array(
                [[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]],
            )
            t_map_to_base = np.array([e, n, u])

            for det in frame_detections:
                p_cam = np.array([det["x_cam"], det["y_cam"], det["z_cam"]])
                p_base = R_base_to_cam @ p_cam + t_base_to_cam
                p_map = R_map_to_base @ p_base + t_map_to_base
                world_observations.append(
                    [
                        p_map[0],
                        p_map[1],
                        p_map[2],
                        det["confidence"],
                        det["instance_id"],
                        float(idx),  # Frame index
                    ],
                )
            t_backproj += time.time() - t0
            processed_count += 1
    except KeyboardInterrupt:
        print(f"\n[Interrupt] Pipeline stopped by user (Ctrl+C). Proceeding to generate results for the {len(world_observations)} observations accumulated so far...")

    world_obs = (
        np.array(world_observations, dtype=np.float32)
        if world_observations
        else np.array([], dtype=np.float32)
    )
    print(f"Total accumulated world observations: {len(world_obs)}")

    # Save world observations (with frame_idx column) to config.BACKPROJ_OUT_DIR
    utils.clear_dir(config.BACKPROJ_OUT_DIR)
    np.save(os.path.join(config.BACKPROJ_OUT_DIR, "world_obs.npy"), world_obs)
    print(f"Saved {len(world_obs)} world observations to {os.path.join(config.BACKPROJ_OUT_DIR, 'world_obs.npy')}")

    # 8. Run Step 5: Multiview Observation Fusion
    method = getattr(config, "FUSION_METHOD", "dbscan")
    label = {"tracking": "Frame-to-Frame Tracking", "voxel": "Voxel Occupancy"}.get(
        method, "DBSCAN Clustering")
    print(f"\n--- STEP 5: Multiview Fusion ({label}) ---")
    t_start = time.time()
    if len(world_obs) > 0:
        if method == "voxel":
            landmarks = process_voxel_fusion(world_obs)
        elif method == "tracking":
            landmarks = process_tracking_fusion(world_obs)
        else:
            landmarks = process_multiview_fusion(
                world_obs,
                eps=config.MULTIVIEW_FUSION_EPS,
                min_samples=config.MULTIVIEW_FUSION_MIN_SAMPLES,
            )
    else:
        landmarks = np.array([], dtype=np.float32)
    t_multi = time.time() - t_start
    print(f"Fused into {len(landmarks)} unique landmarks.")

    redundancy_info = []
    if len(landmarks) > 0:
        obs_counts = landmarks[:, 4]
        avg_obs = float(np.mean(obs_counts))
        max_obs = int(np.max(obs_counts))
        min_obs = int(np.min(obs_counts))

        count_3_5 = int(np.sum((obs_counts >= 3) & (obs_counts <= 5)))
        count_6_10 = int(np.sum((obs_counts >= 6) & (obs_counts <= 10)))
        count_gt_10 = int(np.sum(obs_counts > 10))

        redundancy_info = [
            f"  - Average observations per landmark: {avg_obs:.1f}",
            f"  - Max observations per landmark: {max_obs}",
            f"  - Min observations per landmark: {min_obs}",
            "  - Observation distribution:",
            f"    - Observed 3-5 times: {count_3_5} landmarks",
            f"    - Observed 6-10 times: {count_6_10} landmarks",
            f"    - Observed >10 times: {count_gt_10} landmarks",
        ]
        for line in redundancy_info:
            print(line)

    # 9. Run Step 6: Biology Prior Correction (Corymb Inference)
    print("\n--- STEP 6: Corymb Biology Prior Inference ---")
    t_start = time.time()
    landmarks_bio = process_bio_sanity(
        landmarks,
        expected_per_corymb=config.BIO_SANITY_EXPECTED_FLOWERS_PER_CORYMB,
        intra_spacing=config.BIO_SANITY_INTRA_FLOWER_SPACING_M,
        merge_distance=config.BIO_SANITY_MERGE_DISTANCE_M,
        corymb_radius=config.BIO_SANITY_CORYMB_RADIUS_M,
    )
    t_bio = time.time() - t_start
    print(f"Final landmarks count after inference: {len(landmarks_bio)}")

    # 10. Run Step 7: Tree Assignment
    print("\n--- STEP 7: Tree Assignment ---")
    t_start = time.time()
    per_tree_flowers = process_tree_assignment(
        landmarks_bio,
        tree_map_filepath=config.TREE_MAP_FILE,
        max_radius=config.TREE_ASSIGNMENT_MAX_RADIUS_M,
    )
    t_tree = time.time() - t_start

    # 11. Run Step 8: Thinning Decision & Save Output
    print("\n--- STEP 8: Thinning Decision ---")
    t_start = time.time()
    plan = process_thinning_decision(
        per_tree_flowers,
        target=config.THINNING_TARGET_FLOWERS_PER_TREE,
        agronomic_min=config.THINNING_AGRONOMIC_MIN,
        agronomic_max=config.THINNING_AGRONOMIC_MAX,
    )
    t_decision = time.time() - t_start

    # Save output plan and visualization plot
    utils.clear_dir(config.THINNING_OUT_DIR)
    final_output_path = os.path.join(
        config.THINNING_OUT_DIR,
        "thinning_plan.json",
    )
    with open(final_output_path, "w") as f:
        json.dump(plan, f, indent=4)

    plot_path = os.path.join(config.THINNING_OUT_DIR, "thinning_plan_plot.png")
    utils.save_thinning_plot(plan, plot_path)

    # Save intermediate results to disk so we can recreate plots/maps without re-running full inference
    utils.clear_dir(config.BIO_SANITY_OUT_DIR)
    np.save(os.path.join(config.BIO_SANITY_OUT_DIR, "landmarks_bio.npy"), landmarks_bio)

    utils.clear_dir(config.TREE_ASSIGN_OUT_DIR)
    with open(os.path.join(config.TREE_ASSIGN_OUT_DIR, "per_tree_flowers.json"), "w") as f:
        json.dump(per_tree_flowers, f, indent=4)

    # Save processed range metadata to disk
    ts_start, ts_end = None, None
    if processed_count > 0:
        ts_start = utils.parse_frame_timestamp(left_files[0])
        ts_end = utils.parse_frame_timestamp(left_files[processed_count - 1])

    meta_path = os.path.join(config.OUTPUT_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "processed_frames": processed_count,
            "ts_start": ts_start,
            "ts_end": ts_end
        }, f, indent=4)

    # Save 2D cluster map
    trees = load_tree_map(config.TREE_MAP_FILE)
    map_path = os.path.join(config.THINNING_OUT_DIR, "2d_cluster_map.png")
    utils.save_2d_cluster_map(trees, per_tree_flowers, map_path)

    total_time = time.time() - start_time
    # Keep the input count distinct from successfully completed frames. A run can
    # be interrupted after only a prefix has been processed.
    input_pair_count = len(left_files)
    avg_fps = processed_count / total_time if total_time > 0 else 0.0

    # Write execution report to file
    report_path = os.path.join(config.THINNING_OUT_DIR, "execution_report.log")
    num_frames = processed_count
    
    total_flowers = len(landmarks_bio)
    scanned_trees = sum(1 for item in plan if item["total_count"] > 0)

    report_lines = [
        "==================================================",
        "          OFFLINE PIPELINE EXECUTION REPORT       ",
        "==================================================",
        f"Input Image Pairs:                 {input_pair_count}",
        f"Successfully Processed Frames:     {num_frames}",
        f"Total Flower Landmarks Detected:   {total_flowers}",
        f"Total Trees Scanned:               {scanned_trees}",
        f"Total Execution Time:              {total_time:.3f} seconds",
        f"Average Overall Rate:              {avg_fps:.3f} Hz (FPS)",
        "",
        "Step Execution Breakdown:",
        f"  - Step 1 (Segmentation):   {t_seg:.3f} s (Avg: {num_frames / t_seg if t_seg > 0 else 0:.2f} Hz / FPS)",
        f"  - Step 2 (Disparity):      {t_disp:.3f} s (Avg: {num_frames / t_disp if t_disp > 0 else 0:.2f} Hz / FPS)",
        f"  - Step 3 (Depth Fusion):   {t_fuse:.3f} s (Avg: {num_frames / t_fuse if t_fuse > 0 else 0:.2f} Hz / FPS)",
        f"  - Step 4 (Backprojection): {t_backproj:.3f} s (Avg: {num_frames / t_backproj if t_backproj > 0 else 0:.2f} Hz / FPS)",
        f"  - Step 5 (Multiview Fuse): {t_multi:.3f} s",
    ]
    if redundancy_info:
        report_lines.append("    Clustering Redundancy Analysis:")
        report_lines.extend(["  " + line for line in redundancy_info])

    report_lines.extend(
        [
            f"  - Step 6 (Bio Sanity):     {t_bio:.3f} s",
            f"  - Step 7 (Tree Assign):    {t_tree:.3f} s",
            f"  - Step 8 (Decision):       {t_decision:.3f} s",
            "==================================================",
        ],
    )

    report_text = "\n".join(report_lines)
    with open(report_path, "w") as rf:
        rf.write(report_text)

    print("\n" + report_text)
    print(f"Report saved to: {report_path}")
    print(f"Thinning plot saved to: {plot_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run modular offline apple flower mapping pipeline.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Maximum number of frames to process (-1 for all, default: 10)",
    )
    args = parser.parse_args()

    max_frames = args.max_frames if args.max_frames >= 0 else None
    run_offline_pipeline(max_frames=max_frames)
