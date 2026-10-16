#!/usr/bin/env python3
"""Fuse flower detections into a persistent per-tree map.

Publishes /orchard/flower_map, every localized flower tagged with the tree it
belongs to, plus RViz markers coloured by tree.
"""

from pathlib import Path

import numpy as np
import rclpy
import yaml
from ament_index_python import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from oskar_msgs.msg import (FlowerDetections3D, FlowerLandmark, FlowerLandmarks,
                            PerTreeFlowers)
from oskar_pruning.tracking import FlowerTracker, assign_to_trees

# Fixed hue per tree, assigned in order and never cycled.
TREE_COLOURS = [
    (0.165, 0.471, 0.839),
    (0.922, 0.408, 0.204),
    (0.212, 0.600, 0.404),
    (0.545, 0.353, 0.741),
    (0.851, 0.647, 0.125),
    (0.318, 0.663, 0.745),
]
UNASSIGNED_COLOUR = (0.722, 0.714, 0.682)


class FlowerMapNode(Node):
    def __init__(self):
        super().__init__("flower_map_node")

        self.declare_parameter("tree_map_file", "config/tree_map_sim.yaml")
        self.declare_parameter("tree_assignment_radius_m", 0.8)
        self.declare_parameter("match_radius_m", 0.06)
        self.declare_parameter("max_gap_frames", 2)
        self.declare_parameter("min_observations", 3)
        self.declare_parameter("merge_radius_m", 0.05)
        self.declare_parameter("publish_interval_s", 1.0)
        self.declare_parameter("map_frame", "map")

        self.map_frame = self.get_parameter("map_frame").value
        self.assignment_radius = self.get_parameter("tree_assignment_radius_m").value
        self.trees = self._load_tree_map(self.get_parameter("tree_map_file").value)

        self.tracker = FlowerTracker(
            match_radius=self.get_parameter("match_radius_m").value,
            max_gap_frames=self.get_parameter("max_gap_frames").value,
            min_observations=self.get_parameter("min_observations").value,
            merge_radius=self.get_parameter("merge_radius_m").value,
        )

        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(FlowerDetections3D, "/pruning/detections_3d",
                                 self._detections_callback, qos)
        self.create_subscription(String, "/pruning/scan_event", self._scan_event_callback, qos)
        self.map_pub = self.create_publisher(PerTreeFlowers, "/orchard/flower_map", qos)
        self.marker_pub = self.create_publisher(MarkerArray, "/orchard/flower_markers", qos)
        self.create_service(Trigger, "/pruning/reset_map", self._reset_callback)
        self.create_timer(self.get_parameter("publish_interval_s").value, self._publish)

        self.last_stamp = None
        # Only true while the arm is sweeping a tree.
        self.accepting = False
        self.get_logger().info(f"map ready: {len(self.trees)} trees from the tree map")

    def _load_tree_map(self, path):
        candidate = Path(path)
        if not candidate.is_absolute() and not candidate.exists():
            candidate = Path(get_package_share_directory("oskar_mapping")) / path
        if not candidate.exists():
            raise RuntimeError(f"tree map not found: {path}")
        data = yaml.safe_load(candidate.read_text())
        trees = {int(k): (float(v["x"]), float(v["y"])) for k, v in data["trees"].items()}
        self.get_logger().info(f"tree map: {candidate}")
        return trees

    def _detections_callback(self, msg):
        self.last_stamp = msg.header.stamp
        if not self.accepting:
            # Between scans the arm is swinging into position or driving, and
            # those views are too fast for the tracker to associate; taking them
            # would only fragment flowers into extra landmarks.
            return
        if msg.detections:
            positions = np.array([[d.position.x, d.position.y, d.position.z]
                                  for d in msg.detections])
            confidences = np.array([d.confidence for d in msg.detections])
        else:
            positions = np.empty((0, 3))
            confidences = np.empty(0)
        self.tracker.update(positions, confidences)

    def _scan_event_callback(self, msg):
        """A finished scan flushes open tracks so their flowers reach the map."""
        if msg.data.startswith("scan_sweeping"):
            self.accepting = True
            self.get_logger().info("sweep started, accepting detections")
            return
        if msg.data.startswith("scan_failed"):
            self.accepting = False
            return
        if msg.data.startswith("scan_finished"):
            self.accepting = False
            before = len(self.tracker.landmarks())
            self.tracker.flush()
            after = len(self.tracker.landmarks())
            self.get_logger().info(
                f"scan finished: flushed open tracks, {before} -> {after} landmarks, "
                f"{self.tracker.merge_count} re-observations merged rather than double counted")
            self._publish()

    def _reset_callback(self, request, response):
        self.tracker.reset()
        response.success = True
        response.message = "flower map cleared"
        self.get_logger().info(response.message)
        return response

    def _publish(self):
        landmarks = self.tracker.landmarks()
        ids = self.tracker.landmark_ids()
        tree_ids = assign_to_trees(landmarks[:, :3], self.trees, self.assignment_radius)

        msg = PerTreeFlowers()
        msg.header.stamp = self.last_stamp or self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame

        for tree_id in sorted(self.trees):
            selected = np.nonzero(tree_ids == tree_id)[0]
            group = FlowerLandmarks()
            group.header = msg.header
            group.frame_id = self.map_frame
            for i in selected:
                group.landmarks.append(self._landmark(ids[i], landmarks[i], tree_id))
            msg.tree_ids.append(tree_id)
            msg.confirmed_counts.append(len(selected))
            msg.inferred_counts.append(0)
            msg.per_tree_landmarks.append(group)

        self.map_pub.publish(msg)
        self.marker_pub.publish(self._markers(landmarks, tree_ids, msg.header))

    @staticmethod
    def _landmark(landmark_id, row, tree_id):
        lm = FlowerLandmark()
        lm.landmark_id = int(landmark_id)
        lm.position.x, lm.position.y, lm.position.z = (float(v) for v in row[:3])
        lm.mean_confidence = float(row[3])
        lm.observation_count = int(row[4])
        lm.is_inferred = False
        lm.tree_id = int(tree_id)
        return lm

    def _markers(self, landmarks, tree_ids, header):
        array = MarkerArray()
        marker = Marker()
        marker.header = header
        marker.ns = "flowers"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = 0.03
        marker.pose.orientation.w = 1.0
        ordered = sorted(self.trees)
        for row, tree_id in zip(landmarks, tree_ids):
            from geometry_msgs.msg import Point
            from std_msgs.msg import ColorRGBA
            marker.points.append(Point(x=float(row[0]), y=float(row[1]), z=float(row[2])))
            if tree_id < 0:
                r, g, b = UNASSIGNED_COLOUR
            else:
                r, g, b = TREE_COLOURS[ordered.index(tree_id) % len(TREE_COLOURS)]
            marker.colors.append(ColorRGBA(r=r, g=g, b=b, a=1.0))
        array.markers.append(marker)
        return array


def main(args=None):
    rclpy.init(args=args)
    node = FlowerMapNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
