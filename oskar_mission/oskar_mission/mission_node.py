#!/usr/bin/env python3
"""Drive to each tree that needs thinning, scan it, thin it, move on.

Which trees need thinning is given, from the offline perception. Removal is off
unless thin_trees is set; it needs oskar_thinning_decision and the MoveIt stack
from oskar_removal. Each tree is then thinned in rounds: objects the arm cannot
reach, or failed to remove, are kept, and the decision is made again around
them, until the limit holds or max_rounds runs out.

Positioning uses the simulator's ground-truth pose and the surveyed trunk
positions, the sim counterpart of the RTK tree map in the field. The base moves
in crab mode, so it translates without turning and the arm keeps facing the row
throughout.
"""

from enum import Enum
from pathlib import Path

import numpy as np
import rclpy
from rclpy.signals import SignalHandlerOptions
import tf2_ros
import yaml
from ament_index_python import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String

from oskar_msgs.action import PlanRemovalSequence, RemoveTargets
from oskar_msgs.msg import ThinningCandidates, PerTreeLandmarks, MissionStatus, ScanRequest
from oskar_msgs.srv import DecideThinning
from oskar_removal.conversions import candidates_from_landmarks

MODE_CRAB = 2


class State(Enum):
    WAIT_FOR_POSE = "wait_for_pose"
    DRIVE = "drive"
    SCAN = "scan"
    WAIT_FOR_MAP = "wait_for_map"
    THIN = "thin"
    DONE = "done"


class MissionNode(Node):
    def __init__(self):
        super().__init__("mission_node")

        self.declare_parameter("trees_to_thin", [0, 1])
        self.declare_parameter("tree_map_file", "config/tree_map_sim.yaml")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        # How far back from the trunk the base parks. The arm reaches the canopy
        # from here; closer and the chassis fouls the branches.
        self.declare_parameter("base_gap_m", 2.0)
        # How far behind the trunk, along the row, the base parks. With the
        # trunk level with the arm, the scan line crosses the shoulder axis and
        # a degree of base yaw decides whether the sweep gets past it.
        self.declare_parameter("base_along_row_offset_m", 0.0)
        self.declare_parameter("scan_standoff_m", 0.6)
        self.declare_parameter("scan_z_min", 0.8)
        self.declare_parameter("scan_z_max", 3.9)
        self.declare_parameter("scan_columns", 1)
        self.declare_parameter("position_tolerance_m", 0.05)
        self.declare_parameter("max_speed_mps", 0.6)
        self.declare_parameter("approach_gain", 0.8)
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("thin_trees", False)
        # Upper limit of objects left on a tree; the rest are removed.
        self.declare_parameter("max_kept_per_tree", 20)
        self.declare_parameter("approach_standoff_m", 0.10)
        self.declare_parameter("max_rounds", 4)
        self.declare_parameter("dry_run", False)

        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.base_gap = self.get_parameter("base_gap_m").value
        self.along_row_offset = self.get_parameter("base_along_row_offset_m").value
        self.tolerance = self.get_parameter("position_tolerance_m").value
        self.max_speed = self.get_parameter("max_speed_mps").value
        self.gain = self.get_parameter("approach_gain").value

        self.trees = self._load_tree_map()
        self.queue = [int(t) for t in self.get_parameter("trees_to_thin").value]
        missing = [t for t in self.queue if t not in self.trees]
        if missing:
            raise RuntimeError(f"trees_to_thin names trees not in the tree map: {missing}")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", qos)
        self.mode_pub = self.create_publisher(Int32, "/steering_mode", qos)
        self.scan_pub = self.create_publisher(ScanRequest, "/perception/scan_request", qos)
        self.status_pub = self.create_publisher(MissionStatus, "/mission/status", qos)
        self.create_subscription(String, "/perception/scan_event", self._scan_event_callback, qos)
        self.create_subscription(PerTreeLandmarks, "/orchard/landmark_map",
                                 self._landmark_map_callback, qos)

        self.state = State.WAIT_FOR_POSE
        self.current = None
        self.legs = []
        self.counts = {}
        self.latest_map = None
        self.thin = bool(self.get_parameter("thin_trees").value)
        self.decision_client = self.create_client(DecideThinning, "/decide_thinning")
        self.sequence_client = ActionClient(self, PlanRemovalSequence, "/plan_removal_sequence")
        self.remove_client = ActionClient(self, RemoveTargets, "/remove_targets")
        self.create_timer(1.0 / self.get_parameter("control_rate_hz").value, self._tick)
        self.get_logger().info(f"mission: thin trees {self.queue}")

    def _load_tree_map(self):
        path = Path(self.get_parameter("tree_map_file").value)
        if not path.is_absolute() and not path.exists():
            path = Path(get_package_share_directory("oskar_orchard")) / path
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

    def _landmark_map_callback(self, msg):
        self.counts = dict(zip(msg.tree_ids, msg.confirmed_counts))
        self.latest_map = msg
        if self.state is State.WAIT_FOR_MAP:
            self._start_thinning(msg)

    def _scan_event_callback(self, msg):
        text = msg.data
        if self.state is not State.SCAN or self.current is None:
            return
        if text.startswith("scan_finished") or text.startswith("scan_failed"):
            if text.startswith("scan_failed"):
                self.get_logger().error(f"tree {self.current}: scan failed")
            else:
                self.get_logger().info(
                    f"tree {self.current}: {self.counts.get(self.current, 0)} objects mapped")
            if self.thin and text.startswith("scan_finished"):
                # The map node flushes and republishes on the same event; wait
                # for that map rather than thin from the one before the flush.
                self.state = State.WAIT_FOR_MAP
                return
            self._finish_tree()

    def _finish_tree(self):
        self._publish_status(done=True)
        self.current = None
        self._next_tree()

    # --- thinning --------------------------------------------------------------

    def _start_thinning(self, landmark_map):
        if self.current not in landmark_map.tree_ids:
            self.get_logger().error(f"tree {self.current}: not in the landmark map")
            self._finish_tree()
            return
        landmarks = landmark_map.per_tree_landmarks[landmark_map.tree_ids.index(self.current)]
        header = landmark_map.header
        # Stamp zero: the map frame is fixed, so transforms may use the latest
        # data instead of the scan's stamp, which is long gone by execution.
        header.stamp.sec, header.stamp.nanosec = 0, 0
        self.candidates, skipped = candidates_from_landmarks(
            landmarks.landmarks, self.current, self.trees[self.current], header)
        if skipped:
            self.get_logger().warn(f"tree {self.current}: {skipped} objects without an approach axis")
        self.keep_ids, self.removed_ids = set(), set()
        self.rounds = 0
        self.state = State.THIN
        self._plan_round()

    def _remaining(self):
        remaining = ThinningCandidates()
        remaining.header = self.candidates.header
        remaining.tree_id = self.candidates.tree_id
        remaining.candidates = [c for c in self.candidates.candidates
                                if c.id not in self.removed_ids]
        return remaining

    def _plan_round(self):
        limit = int(self.get_parameter("max_kept_per_tree").value)
        remaining = self._remaining()
        if len(remaining.candidates) <= limit:
            self.get_logger().info(
                f"tree {self.current}: {len(remaining.candidates)} left, limit {limit}, done")
            self._finish_tree()
            return
        if self.rounds >= int(self.get_parameter("max_rounds").value):
            self.get_logger().warn(
                f"tree {self.current}: {len(remaining.candidates)} left after "
                f"{self.rounds} rounds, above the limit of {limit}")
            self._finish_tree()
            return
        self.rounds += 1
        if not self.decision_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("decide_thinning is unavailable; is the thinning stack up?")
            self._finish_tree()
            return
        request = DecideThinning.Request()
        request.candidates = remaining
        request.max_kept = limit
        request.keep_ids = sorted(self.keep_ids)
        self.decision_client.call_async(request).add_done_callback(self._on_decision)

    def _on_decision(self, future):
        decision = future.result().decision
        if not decision.remove_ids:
            self._finish_tree()
            return
        remove = set(decision.remove_ids)
        goal = PlanRemovalSequence.Goal()
        goal.candidates = self._remaining()
        goal.candidates.candidates = [c for c in goal.candidates.candidates if c.id in remove]
        goal.standoff = float(self.get_parameter("approach_standoff_m").value)
        goal.check_reachability = True
        self._send(self.sequence_client, goal, self._on_sequence)

    def _on_sequence(self, result):
        if result is None or not result.success:
            self.get_logger().error(f"tree {self.current}: removal planning failed")
            self._finish_tree()
            return
        # Unreachable objects stay, so the next round chooses around them.
        self.keep_ids.update(result.rejected_ids)
        if result.rejected_ids:
            self.get_logger().info(
                f"tree {self.current}: {len(result.rejected_ids)} unreachable, kept")
        if not result.sequence.targets:
            self._plan_round()
            return
        goal = RemoveTargets.Goal()
        goal.sequence = result.sequence
        goal.dry_run = bool(self.get_parameter("dry_run").value)
        self._send(self.remove_client, goal, self._on_removed)

    def _on_removed(self, result):
        if result is None:
            self.get_logger().error(f"tree {self.current}: removal did not run")
            self._finish_tree()
            return
        for r in result.results:
            (self.removed_ids if r.success else self.keep_ids).add(r.candidate_id)
        self.counts[self.current] = len(self.candidates.candidates) - len(self.removed_ids)
        self._plan_round()

    def _send(self, client, goal, on_result):
        if not client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("removal action server unavailable")
            on_result(None)
            return

        def accepted(future):
            handle = future.result()
            if handle is None or not handle.accepted:
                on_result(None)
                return
            handle.get_result_async().add_done_callback(
                lambda done: on_result(done.result().result))

        client.send_goal_async(goal).add_done_callback(accepted)

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
        target = np.array([trunk[0] - self.along_row_offset, trunk[1] - self.base_gap])
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
        elif self.state in (State.SCAN, State.WAIT_FOR_MAP, State.THIN):
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
        status = MissionStatus()
        status.header.stamp = self.get_clock().now().to_msg()
        status.tree_id = int(self.current)
        status.current_count = int(self.counts.get(self.current, 0))
        status.target_count = (int(self.get_parameter("max_kept_per_tree").value)
                               if self.thin else 0)
        status.done = done
        self.status_pub.publish(status)


def main(args=None):
    # rclpy's own SIGINT handler shuts the context down before the stop below
    # could be published, and Isaac keeps driving on the last /cmd_vel.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = MissionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
