#!/usr/bin/env python3
"""Drive to each tree that needs pruning, scan it, move on.

Steps 1 to 3 of the pruning sequence. Which trees need pruning is given, not
decided here; the density rule that picks flowers to remove is a later work
package, as is the approach and the gripper.

Positioning uses the simulator's ground-truth pose and the surveyed trunk
positions, the sim counterpart of the RTK tree map in the field. The base moves
in crab mode, so it translates without turning and the arm keeps facing the row
throughout.
"""

from enum import Enum
from pathlib import Path

import numpy as np
import rclpy
import tf2_ros
import yaml
from ament_index_python import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String

from oskar_msgs.msg import PerTreeFlowers, PruningStatus, ScanRequest

MODE_CRAB = 2


class State(Enum):
    WAIT_FOR_POSE = "wait_for_pose"
    DRIVE = "drive"
    SCAN = "scan"
    DONE = "done"


class MissionNode(Node):
    def __init__(self):
        super().__init__("mission_node")

        self.declare_parameter("trees_to_prune", [0, 1])
        self.declare_parameter("tree_map_file", "config/tree_map_sim.yaml")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        # How far back from the trunk the base parks. The arm reaches the canopy
        # from here; closer and the chassis fouls the branches.
        self.declare_parameter("base_gap_m", 2.0)
        self.declare_parameter("scan_standoff_m", 0.6)
        self.declare_parameter("scan_z_min", 0.8)
        self.declare_parameter("scan_z_max", 3.9)
        self.declare_parameter("scan_columns", 1)
        self.declare_parameter("position_tolerance_m", 0.05)
        self.declare_parameter("max_speed_mps", 0.6)
        self.declare_parameter("approach_gain", 0.8)
        self.declare_parameter("control_rate_hz", 20.0)

        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.base_gap = self.get_parameter("base_gap_m").value
        self.tolerance = self.get_parameter("position_tolerance_m").value
        self.max_speed = self.get_parameter("max_speed_mps").value
        self.gain = self.get_parameter("approach_gain").value

        self.trees = self._load_tree_map()
        self.queue = [int(t) for t in self.get_parameter("trees_to_prune").value]
        missing = [t for t in self.queue if t not in self.trees]
        if missing:
            raise RuntimeError(f"trees_to_prune names trees not in the tree map: {missing}")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", qos)
        self.mode_pub = self.create_publisher(Int32, "/steering_mode", qos)
        self.scan_pub = self.create_publisher(ScanRequest, "/pruning/scan_request", qos)
        self.status_pub = self.create_publisher(PruningStatus, "/pruning/status", qos)
        self.create_subscription(String, "/pruning/scan_event", self._scan_event_callback, qos)
        self.create_subscription(PerTreeFlowers, "/orchard/flower_map",
                                 self._flower_map_callback, qos)

        self.state = State.WAIT_FOR_POSE
        self.current = None
        self.legs = []
        self.counts = {}
        self.create_timer(1.0 / self.get_parameter("control_rate_hz").value, self._tick)
        self.get_logger().info(f"mission: prune trees {self.queue}")

    def _load_tree_map(self):
        path = Path(self.get_parameter("tree_map_file").value)
        if not path.is_absolute() and not path.exists():
            path = Path(get_package_share_directory("oskar_mapping")) / path
        data = yaml.safe_load(path.read_text())
        return {int(k): np.array([float(v["x"]), float(v["y"])]) for k, v in data["trees"].items()}

    def _pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except tf2_ros.TransformException:
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return np.array([t.x, t.y]), float(yaw)

    def _flower_map_callback(self, msg):
        self.counts = dict(zip(msg.tree_ids, msg.confirmed_counts))

    def _scan_event_callback(self, msg):
        text = msg.data
        if self.state is not State.SCAN or self.current is None:
            return
        if text.startswith("scan_finished") or text.startswith("scan_failed"):
            if text.startswith("scan_failed"):
                self.get_logger().error(f"tree {self.current}: scan failed")
            else:
                self.get_logger().info(
                    f"tree {self.current}: {self.counts.get(self.current, 0)} flowers mapped")
            self._publish_status(done=True)
            self.current = None
            self._next_tree()

    def _next_tree(self):
        self._stop()
        if not self.queue:
            self.state = State.DONE
            self.get_logger().info("all trees scanned")
            return
        self.current = self.queue.pop(0)
        # Two straight legs rather than a diagonal: crab mode steers all wheels
        # to the direction of travel, and a constant direction per leg means the
        # wheels are steered once instead of chasing a turning error vector.
        trunk = self.trees[self.current]
        target = np.array([trunk[0], trunk[1] - self.base_gap])
        self.legs = [np.array([None, target[1]]), target]
        self.state = State.DRIVE
        self.get_logger().info(
            f"tree {self.current}: driving to ({target[0]:.2f}, {target[1]:.2f})")

    def _tick(self):
        pose = self._pose()
        if pose is None:
            if self.state is not State.WAIT_FOR_POSE:
                self.get_logger().warn("lost the base pose", throttle_duration_sec=5.0)
            return
        if self.state is State.WAIT_FOR_POSE:
            self.get_logger().info("base pose acquired")
            self._next_tree()
            return
        if self.state is State.DRIVE:
            self._drive(pose)
        elif self.state is State.SCAN:
            self._publish_status(done=False)

    def _drive(self, pose):
        position, yaw = pose
        leg = self.legs[0]
        error = np.array([
            0.0 if leg[0] is None else float(leg[0]) - position[0],
            0.0 if leg[1] is None else float(leg[1]) - position[1],
        ])
        if np.linalg.norm(error) < self.tolerance:
            self.legs.pop(0)
            self._stop()
            if self.legs:
                return
            self._start_scan()
            return

        velocity = np.clip(self.gain * error, -self.max_speed, self.max_speed)
        c, s = np.cos(-yaw), np.sin(-yaw)
        cmd = Twist()
        cmd.linear.x = float(c * velocity[0] - s * velocity[1])
        cmd.linear.y = float(s * velocity[0] + c * velocity[1])
        self.mode_pub.publish(Int32(data=MODE_CRAB))
        self.cmd_pub.publish(cmd)

    def _start_scan(self):
        self.state = State.SCAN
        request = ScanRequest()
        request.header.stamp = self.get_clock().now().to_msg()
        request.tree_id = int(self.current)
        request.standoff_m = float(self.get_parameter("scan_standoff_m").value)
        request.z_min = float(self.get_parameter("scan_z_min").value)
        request.z_max = float(self.get_parameter("scan_z_max").value)
        request.columns = int(self.get_parameter("scan_columns").value)
        self.scan_pub.publish(request)
        self.get_logger().info(f"tree {self.current}: in position, scanning")

    def _stop(self):
        self.mode_pub.publish(Int32(data=MODE_CRAB))
        self.cmd_pub.publish(Twist())

    def _publish_status(self, done):
        if self.current is None:
            return
        status = PruningStatus()
        status.header.stamp = self.get_clock().now().to_msg()
        status.tree_id = int(self.current)
        status.current_count = int(self.counts.get(self.current, 0))
        status.target_count = 0
        status.done = done
        self.status_pub.publish(status)


def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
