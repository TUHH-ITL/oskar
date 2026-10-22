#!/usr/bin/env python3
"""Sweep the wrist camera over a tree so every flower is seen.

The camera path is planned in the map frame, converted to joint angles with the
UR20 kinematics, and streamed to Isaac as joint position targets. The steps are
small on purpose: the tracker associates a flower with its previous sighting
only within a few centimetres, so the camera has to move less than that between
frames or every flower fragments into separate landmarks.

This bypasses MoveIt, which the scene is otherwise wired for through
topic_based_ros2_control. That path is the one for real hardware, where the arm
has to avoid the tree it is reaching into. Here the camera stays outside the
canopy on a path that is clear by construction, and going direct removes the
whole controller stack from the loop while the perception is being brought up.
Do not run this and a ros2_control chain at the same time: both drive
/isaac_joint_commands. With the removal stack up, set command_mode to
"trajectory": the same joint path then goes to the arm's joint trajectory
controller, which is the only thing writing the arm commands.
"""

import numpy as np
import rclpy
import tf2_ros
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

from oskar_msgs.msg import ScanRequest
from oskar_online_perception.kinematics import (JOINT_NAMES, forward_kinematics,
                                      inverse_kinematics, transform)

# Camera optical axis pointing along +y of the map, image right along +x,
# image down along -z: the pose that faces a tree row lying across +y.
LOOK_ALONG_PLUS_Y = np.column_stack([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


class ScanControllerNode(Node):
    def __init__(self):
        super().__init__("scan_controller_node")

        self.declare_parameter("joint_command_topic", "/isaac_joint_commands")
        self.declare_parameter("joint_state_topic", "/isaac_joint_states")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("tool_translation", [-0.085, 0.0599, 0.035])
        self.declare_parameter("tool_rotation_xyzw", [0.092296, -0.092296, 0.701057, -0.701057])
        self.declare_parameter("command_rate_hz", 30.0)
        # Camera speed along the path. The tracker's match radius is 0.06 m, so
        # at 30 Hz this leaves a wide margin against fragmenting a flower.
        self.declare_parameter("scan_speed_mps", 0.12)
        self.declare_parameter("path_step_m", 0.01)
        self.declare_parameter("column_spacing_m", 0.35)
        # Upper bound on the swing into position; normally the arm arrives sooner
        # and the sweep starts then.
        self.declare_parameter("settle_s", 1.0)
        self.declare_parameter("arrival_tolerance_rad", 0.01)
        # A step larger than this between waypoints millimetres apart is a
        # different arm configuration, not motion along the path.
        self.declare_parameter("max_joint_step_rad", 0.15)
        # "joint_commands" streams positions to Isaac directly; "trajectory"
        # sends the path to a FollowJointTrajectory controller instead.
        self.declare_parameter("command_mode", "joint_commands")
        self.declare_parameter("trajectory_action",
                               "/ur_manipulator_controller/follow_joint_trajectory")
        # Speed of the swing to the start of the sweep in trajectory mode.
        self.declare_parameter("move_to_start_speed_rad_s", 0.4)

        self.command_rate = self.get_parameter("command_rate_hz").value
        self.scan_speed = self.get_parameter("scan_speed_mps").value
        self.path_step = self.get_parameter("path_step_m").value
        self.column_spacing = self.get_parameter("column_spacing_m").value
        self.arrival_tolerance = self.get_parameter("arrival_tolerance_rad").value
        self.max_joint_step = self.get_parameter("max_joint_step_rad").value
        self.base_frame = self.get_parameter("base_frame").value
        self.map_frame = self.get_parameter("map_frame").value
        self.tool = self._tool_transform()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(JointState, self.get_parameter("joint_state_topic").value,
                                 self._joint_state_callback, qos)
        self.create_subscription(ScanRequest, "/perception/scan_request", self._request_callback, qos)
        self.command_pub = self.create_publisher(
            JointState, self.get_parameter("joint_command_topic").value, qos)
        self.event_pub = self.create_publisher(String, "/perception/scan_event", qos)
        self.command_mode = self.get_parameter("command_mode").value
        if self.command_mode not in ("joint_commands", "trajectory"):
            raise RuntimeError(f"unknown command_mode {self.command_mode}")
        self.trajectory_client = ActionClient(
            self, FollowJointTrajectory, self.get_parameter("trajectory_action").value)

        self.joints = None
        self.trajectory = None
        self.index = 0
        self.tree_id = None
        self.hold_ticks = 0
        self.sweep_started = 0.0
        self.create_timer(1.0 / self.command_rate, self._tick)
        self.get_logger().info("scan controller ready")

    def _tool_transform(self):
        t = np.array(self.get_parameter("tool_translation").value, dtype=float)
        x, y, z, w = self.get_parameter("tool_rotation_xyzw").value
        T = np.eye(4)
        T[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
        T[:3, 3] = t
        return T

    def _joint_state_callback(self, msg):
        index = {name: i for i, name in enumerate(msg.name)}
        if not all(name in index for name in JOINT_NAMES):
            return
        self.joints = np.array([msg.position[index[name]] for name in JOINT_NAMES])

    def _request_callback(self, request):
        if self.joints is None:
            self.get_logger().warn("no joint state yet, ignoring scan request")
            return
        if self.trajectory is not None:
            self.get_logger().warn(f"already scanning tree {self.tree_id}, ignoring request")
            return

        base = self._base_pose()
        if base is None:
            self.get_logger().warn("no base pose, ignoring scan request")
            return

        trunk = self._trunk_in_base(request, base)
        if trunk is None:
            return

        waypoints = self._plan(trunk, request)
        trajectory, reached, flips = self._solve(waypoints)
        if len(trajectory) < 2:
            self.get_logger().error(
                f"tree {request.tree_id}: no reachable scan path at standoff "
                f"{request.standoff_m:.2f} m")
            self._publish_event(f"scan_failed {request.tree_id}")
            return

        self.trajectory = trajectory
        self.index = 0
        self.tree_id = int(request.tree_id)
        self.hold_ticks = int(self.get_parameter("settle_s").value * self.command_rate)
        # reached is in base_link; report it in the map frame the caller asked in.
        base_z = base[0][2]
        covered = f"{reached[0] + base_z:.2f}..{reached[1] + base_z:.2f}"
        steps = [np.max(np.abs(b - a)) for a, b in zip(trajectory, trajectory[1:])]
        worst = float(np.max(steps)) if steps else 0.0
        duration = len(trajectory) * self.path_step / self.scan_speed
        self.get_logger().info(
            f"tree {self.tree_id}: scanning map z {covered} m of the requested "
            f"{request.z_min:.2f}..{request.z_max:.2f}, over {len(trajectory)} steps "
            f"({duration:.0f} s of simulation at {self.scan_speed:.2f} m/s), "
            f"{len(waypoints) - len(trajectory) - flips} waypoints out of reach, "
            f"{flips} skipped as configuration flips, "
            f"largest joint step between waypoints {worst:.3f} rad")
        self._publish_event(f"scan_started {self.tree_id}")
        if self.command_mode == "trajectory":
            self._send_trajectory([self.trajectory[0]], self._start_duration(), self._on_at_start)

    def _base_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(f"base pose lookup failed: {exc}")
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        return np.array([t.x, t.y, t.z]), float(yaw)

    def _trunk_in_base(self, request, base):
        """Scan target expressed in base_link, from the tree map via TF."""
        position, yaw = base
        if not hasattr(self, "_tree_positions"):
            self.get_logger().error("no tree positions; set the tree_positions parameter")
            return None
        if request.tree_id not in self._tree_positions:
            self.get_logger().error(f"tree {request.tree_id} is not in the tree map")
            return None
        trunk = np.array(self._tree_positions[request.tree_id])
        delta = trunk - position[:2]
        c, s = np.cos(-yaw), np.sin(-yaw)
        return np.array([c * delta[0] - s * delta[1], s * delta[0] + c * delta[1]]), position[2]

    def _plan(self, trunk, request):
        """Camera waypoints in base_link, sweeping up the near face of the tree."""
        trunk_xy, base_z = trunk
        # Approach from whichever side of the trunk the robot already is.
        side = -1.0 if trunk_xy[1] >= 0.0 else 1.0
        y = trunk_xy[1] + side * request.standoff_m

        z_min = request.z_min - base_z
        z_max = request.z_max - base_z
        columns = max(1, int(request.columns))
        offsets = (np.arange(columns) - (columns - 1) / 2.0) * self.column_spacing

        waypoints = []
        for i, offset in enumerate(offsets):
            steps = max(2, int(abs(z_max - z_min) / self.path_step))
            zs = np.linspace(z_min, z_max, steps)
            # Alternate direction so the arm does not fly back between columns.
            if i % 2:
                zs = zs[::-1]
            for z in zs:
                waypoints.append(np.array([trunk_xy[0] + offset, y, z]))
        return waypoints

    # Fallbacks for when continuing from the previous pose fails. Damped least
    # squares converges to whatever branch its seed is nearest, so a single
    # seed walked along the path gets stuck and reports reachable points as
    # unreachable; measured, that cost the top and bottom of the sweep.
    RESTART_SEEDS = (
        (0.0, -0.6, 1.2, -0.6, -1.57, 0.0),
        (0.0, -1.2, 1.8, -0.6, -1.57, 0.0),
        (0.0, -0.3, 0.6, -0.3, -1.57, 0.0),
        (0.0, -2.0, 2.2, -0.2, -1.57, 0.0),
        (0.0, -1.5, -1.5, 0.5, -1.57, 0.0),
    )

    def _solve(self, waypoints):
        """Joint angles for the reachable waypoints, seeded along the path."""
        rotation = LOOK_ALONG_PLUS_Y.copy()
        if waypoints and waypoints[0][1] < 0:
            # Facing -y instead: flip the optical axis and the image right vector.
            rotation = np.column_stack([[-1.0, 0, 0], [0, 0, -1.0], [0, -1.0, 0]])

        seed = self.joints.copy()
        trajectory, reached, flips = [], [], 0
        for point in waypoints:
            target = np.eye(4)
            target[:3, :3] = rotation
            target[:3, 3] = point
            solution, converged, error = inverse_kinematics(
                target, seed, tool=self.tool, position_tolerance=2e-3)
            if not converged or error > 5e-3:
                for restart in self.RESTART_SEEDS:
                    solution, converged, error = inverse_kinematics(
                        target, np.array(restart), tool=self.tool, position_tolerance=2e-3)
                    if converged and error <= 5e-3:
                        break
            if not converged or error > 5e-3:
                continue
            # Neighbouring waypoints are millimetres apart, so a large jump in
            # joint space means a different arm configuration reaching the same
            # point. Following it swings the camera metres between frames and
            # the tracker loses every flower; skipping the waypoint costs only
            # that slice of the sweep.
            if trajectory is not None and len(trajectory):
                if float(np.max(np.abs(solution - trajectory[-1]))) > self.max_joint_step:
                    flips += 1
                    continue
            trajectory.append(solution)
            reached.append(point[2])
            seed = solution
        span = (min(reached), max(reached)) if reached else (0.0, 0.0)
        return trajectory, span, flips

    def _start_duration(self):
        travel = float(np.max(np.abs(self.trajectory[0] - self.joints)))
        speed = self.get_parameter("move_to_start_speed_rad_s").value
        return max(2.0, travel / speed)

    def _send_trajectory(self, points, first_time, on_done):
        """Points at first_time, then one path step of travel apart."""
        if not self.trajectory_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("joint trajectory controller is unavailable")
            self._abort_scan()
            return
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(JOINT_NAMES)
        step = self.path_step / self.scan_speed
        for i, joints in enumerate(points):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in joints]
            t = first_time + i * step
            point.time_from_start = DurationMsg(sec=int(t), nanosec=int((t % 1.0) * 1e9))
            goal.trajectory.points.append(point)
        self.trajectory_client.send_goal_async(goal).add_done_callback(
            lambda future: self._on_goal_response(future, on_done))

    def _on_goal_response(self, future, on_done):
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error("joint trajectory goal rejected")
            self._abort_scan()
            return
        handle.get_result_async().add_done_callback(
            lambda result: on_done(result.result().result.error_code))

    def _on_at_start(self, error_code):
        if error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().error(f"move to the sweep start failed, error {error_code}")
            self._abort_scan()
            return
        self._publish_event(f"scan_sweeping {self.tree_id}")
        self._send_trajectory(self.trajectory, self.path_step / self.scan_speed,
                              self._on_swept)

    def _on_swept(self, error_code):
        if error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().error(f"sweep failed, error {error_code}")
            self._abort_scan()
            return
        self.get_logger().info(f"tree {self.tree_id}: scan complete")
        self._publish_event(f"scan_finished {self.tree_id}")
        self.trajectory = None
        self.tree_id = None

    def _abort_scan(self):
        self._publish_event(f"scan_failed {self.tree_id}")
        self.trajectory = None
        self.tree_id = None

    def _tick(self):
        if self.trajectory is None or self.command_mode != "joint_commands":
            return
        if self.hold_ticks > 0:
            # The arm swings from wherever it was to the start of the sweep at
            # full speed, which moves the camera far more per frame than the
            # tracker can associate. Wait for it to actually arrive, not for a
            # fixed time, and only then tell the map to start listening.
            self.hold_ticks -= 1
            self._send(self.trajectory[0])
            if self.joints is not None:
                error = float(np.max(np.abs(self.joints - self.trajectory[0])))
                if error < self.arrival_tolerance:
                    self.hold_ticks = 0
            if self.hold_ticks == 0:
                self.sweep_started = self.get_clock().now().nanoseconds * 1e-9
                self._publish_event(f"scan_sweeping {self.tree_id}")
            return

        # Position along the path from elapsed time, so the commanded speed is
        # the requested speed whatever rate the timer actually fires at. Counting
        # ticks instead ties the arm's speed to the clock, and the timer here
        # runs far below its nominal rate.
        now = self.get_clock().now().nanoseconds * 1e-9
        travelled = max(0.0, (now - self.sweep_started)) * self.scan_speed
        # Only ever advance. Simulation time can step backwards -- the clock
        # resets on stop, and a gap in /clock reads as zero -- and a negative
        # index silently walks the path from the far end before running off it.
        self.index = max(self.index, int(travelled / self.path_step))
        if self.index >= len(self.trajectory):
            self._send(self.trajectory[-1])
            self.get_logger().info(f"tree {self.tree_id}: scan complete")
            self._publish_event(f"scan_finished {self.tree_id}")
            self.trajectory = None
            self.tree_id = None
            return
        self._send(self.trajectory[self.index])

    def _send(self, joints):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES)
        msg.position = [float(v) for v in joints]
        self.command_pub.publish(msg)

    def _publish_event(self, text):
        self.event_pub.publish(String(data=text))

    def set_tree_positions(self, positions):
        self._tree_positions = positions


def main(args=None):
    rclpy.init(args=args)
    node = ScanControllerNode()
    node.declare_parameter("tree_map_file", "config/tree_map_sim.yaml")

    from pathlib import Path
    import yaml
    from ament_index_python import get_package_share_directory

    path = Path(node.get_parameter("tree_map_file").value)
    if not path.is_absolute() and not path.exists():
        path = Path(get_package_share_directory("oskar_orchard")) / path
    data = yaml.safe_load(path.read_text())
    node.set_tree_positions({int(k): (float(v["x"]), float(v["y"]))
                             for k, v in data["trees"].items()})
    node.get_logger().info(f"tree map: {path}")

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
