#!/usr/bin/env python3
"""Score the published flower map against the simulator's own flower prims.

The scene carries every flower as a named prim, so recall, precision and 3D
error are measurable here in a way they never were on the real orchard, where
only per-tree hand counts existed and only for fourteen trees.

Matching is one-to-one within a radius, by nearest pair first, so one detection
cannot claim several ground-truth flowers and inflate recall.
"""

import json
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy.optimize import linear_sum_assignment
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from oskar_msgs.msg import PerTreeLandmarks


class GroundTruthEvalNode(Node):
    def __init__(self):
        super().__init__("gt_eval_node")

        self.declare_parameter("ground_truth_file", "config/orchard_gt_sim.json")
        self.declare_parameter("match_radius_m", 0.10)
        self.declare_parameter("report_on_scan_end", True)
        self.declare_parameter("report_interval_s", 10.0)
        # Flowers the camera never had a line to cannot be scored as misses.
        self.declare_parameter("evaluate_z_min", 0.0)
        self.declare_parameter("evaluate_z_max", 100.0)
        # Trees to score. Empty means every tree that has landmarks, so a tree
        # nobody has driven to yet is not counted as a tree of pure misses.
        self.declare_parameter("evaluate_trees", [])
        # A corymb counts as found if any of its flowers was. This is the unit
        # the field work counts in, where a "Blütenbüschel" is one observation.
        self.declare_parameter("report_corymbs", True)

        self.match_radius = self.get_parameter("match_radius_m").value
        self.z_min = self.get_parameter("evaluate_z_min").value
        self.z_max = self.get_parameter("evaluate_z_max").value
        self.truth = self._load()

        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(PerTreeLandmarks, "/orchard/landmark_map", self._map_callback, qos)
        if self.get_parameter("report_on_scan_end").value:
            self.create_subscription(String, "/perception/scan_event", self._event_callback, qos)
        self.create_timer(self.get_parameter("report_interval_s").value, self._report)
        # Where the misses are is the part a table cannot show: whether they sit
        # high in the canopy, on the far side, or scattered through it.
        self.marker_pub = self.create_publisher(MarkerArray, "/orchard/eval_markers", qos)

        self.latest = None
        self.get_logger().info(
            f"evaluating against {sum(len(v) for v in self.truth.values())} flowers "
            f"in {len(self.truth)} trees, match radius {self.match_radius:.2f} m")

    def _load(self):
        path = Path(self.get_parameter("ground_truth_file").value)
        if not path.is_absolute() and not path.exists():
            from ament_index_python import get_package_share_directory
            path = Path(get_package_share_directory("oskar_orchard")) / path
        data = json.loads(path.read_text())
        self.corymbs = {int(k): np.array([f["corymb"] for f in v["flowers"]])
                        for k, v in data["trees"].items()}
        return {int(k): np.array([f["position"] for f in v["flowers"]])
                for k, v in data["trees"].items()}

    def _map_callback(self, msg):
        self.latest = msg

    def _event_callback(self, msg):
        if msg.data.startswith("scan_finished"):
            self._report()

    def _report(self):
        if self.latest is None:
            return
        wanted = [int(t) for t in self.get_parameter("evaluate_trees").value]
        rows = []
        marked = {"matched": [], "missed": [], "false positive": []}
        total = np.zeros(4)  # matched, detected, truth, error sum
        corymb_total = np.zeros(2)  # found, total
        for tree_id, group in zip(self.latest.tree_ids, self.latest.per_tree_landmarks):
            tree_id = int(tree_id)
            truth = self.truth.get(tree_id)
            if truth is None:
                continue
            if wanted and tree_id not in wanted:
                continue
            if not wanted and not group.landmarks:
                continue
            in_band = (truth[:, 2] >= self.z_min) & (truth[:, 2] <= self.z_max)
            corymbs = self.corymbs[tree_id][in_band]
            truth = truth[in_band]
            found = np.array([[lm.position.x, lm.position.y, lm.position.z]
                              for lm in group.landmarks]).reshape(-1, 3)
            matched, errors, pairs, hits = self._match(found, truth)
            missed = np.setdiff1d(np.arange(len(truth)), pairs)
            spurious = np.setdiff1d(np.arange(len(found)), hits)
            marked["matched"].extend(truth[pairs])
            marked["missed"].extend(truth[missed])
            marked["false positive"].extend(found[spurious])
            recall = matched / len(truth) if len(truth) else float("nan")
            precision = matched / len(found) if len(found) else float("nan")
            rows.append((tree_id, len(truth), len(found), matched, recall, precision,
                         float(np.mean(errors)) if errors.size else float("nan")))
            total += [matched, len(found), len(truth), errors.sum()]
            if len(corymbs):
                corymb_total += [len(set(corymbs[pairs])), len(set(corymbs))]

        if not rows:
            return
        self._publish_markers(marked)
        lines = ["flower map against the simulator's flower prims",
                 "  tree   truth  mapped  matched   recall  precision   mean error"]
        for tree_id, n_truth, n_found, matched, recall, precision, error in rows:
            lines.append(f"  {tree_id:>4} {n_truth:>7} {n_found:>7} {matched:>8} "
                         f"{recall:8.1%} {precision:10.1%} {error * 100:9.1f} cm")
        matched, detected, truth_count, error_sum = total
        lines.append(f"  all  {truth_count:>7.0f} {detected:>7.0f} {matched:>8.0f} "
                     f"{matched / truth_count if truth_count else 0:8.1%} "
                     f"{matched / detected if detected else 0:10.1%} "
                     f"{error_sum / matched * 100 if matched else 0:9.1f} cm")
        if self.get_parameter("report_corymbs").value and corymb_total[1]:
            lines.append(f"  corymbs reached: {corymb_total[0]:.0f} of {corymb_total[1]:.0f} "
                         f"({corymb_total[0] / corymb_total[1]:.1%}) -- the unit the field "
                         f"counts in")
        self.get_logger().info("\n".join(lines))

    def _match(self, found, truth):
        """One-to-one nearest matching inside the radius.

        Returns the match count, the matched distances, and which ground-truth
        flowers and which landmarks were matched.
        """
        empty = np.array([], dtype=int)
        if len(found) == 0 or len(truth) == 0:
            return 0, np.array([]), empty, empty
        cost = np.linalg.norm(found[:, None, :] - truth[None, :, :], axis=2)
        # Pairs beyond the radius must never be chosen, so price them out of the
        # assignment rather than trimming the matrix.
        blocked = cost > self.match_radius
        cost = np.where(blocked, 1e6, cost)
        rows, cols = linear_sum_assignment(cost)
        keep = ~blocked[rows, cols]
        return int(keep.sum()), cost[rows[keep], cols[keep]], cols[keep], rows[keep]

    def _publish_markers(self, marked):
        colours = {"matched": (0.1, 0.9, 0.2), "missed": (0.9, 0.1, 0.1),
                   "false positive": (1.0, 0.8, 0.0)}
        array = MarkerArray()
        for index, (name, positions) in enumerate(marked.items()):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = name
            marker.id = index
            marker.type = Marker.SPHERE_LIST
            marker.action = Marker.ADD
            marker.scale.x = marker.scale.y = marker.scale.z = 0.03
            marker.color.r, marker.color.g, marker.color.b = colours[name]
            marker.color.a = 1.0
            marker.pose.orientation.w = 1.0
            for position in positions:
                point = Point()
                point.x, point.y, point.z = (float(v) for v in position)
                marker.points.append(point)
            array.markers.append(marker)
        self.marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthEvalNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
