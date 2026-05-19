#!/usr/bin/env python3
"""4-Wheel Steering controller for the Tipard robot in Isaac Sim.
ROS 2 Humble integration via rclpy (background thread).

Robot dimensions (from tipard/tipard.urdf joint origins):
  Wheelbase         L = 3.000 m   (front axle x=+1.463, rear axle x=-1.537)
  Steering track    T = 2.058 m   (wheel-centre to wheel-centre laterally)
  Wheel radius      r = 0.349 m

Articulation DOFs (8 total):
  Steering (position targets): front_right_leg, front_left_leg,
                                back_right_leg,  back_left
  Drive    (velocity targets): front_right_wheel, front_left_wheel,
                                back_right_wheel,  back_left_wheel

ROS 2 topics:
  Subscribe  /cmd_vel        geometry_msgs/Twist
             /four_ws_mode   std_msgs/Int32    (1=opposite-phase, 2=in-phase, 3=pivot)
  Publish    /joint_states   sensor_msgs/JointState  (via Isaac Sim ROS 2 bridge OmniGraph)

Usage:
  python four_ws_isaac_sim.py [--headless]
"""

import argparse
import math
import os
import re
import tempfile
import threading

import numpy as np

# ---------------------------------------------------------------------------
# 1.  Isaac Sim bootstrap – must happen before any omni.* imports
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Tipard 4WS Isaac Sim")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
args, _ = parser.parse_known_args()

from omni.isaac.kit import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless})

# ---------------------------------------------------------------------------
# 2.  Isaac / Omniverse imports (must follow SimulationApp construction)
# ---------------------------------------------------------------------------
import carb  # noqa: E402
import omni.kit.commands  # noqa: E402

# ---------------------------------------------------------------------------
# 3.  ROS 2
# ---------------------------------------------------------------------------
import rclpy  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from omni.importer.urdf import _urdf  # noqa: E402
from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.robots import Robot  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.time import Time  # noqa: E402
from std_msgs.msg import Int32  # noqa: E402

# ---------------------------------------------------------------------------
# 4.  Robot geometry constants
# ---------------------------------------------------------------------------
WHEELBASE = 3.000  # m – front-to-rear axle distance
WHEEL_TRACK = 2.100  # m – lateral distance between steering joint centres
DIST_STEER_TO_WHEEL = (
    0.021  # m – lateral offset from steering joint to wheel contact
)
#     (tipard: front_right_wheel joint y = -0.0208 m)
WHEEL_RADIUS = 0.349  # m
STEERING_TRACK = WHEEL_TRACK - 2.0 * DIST_STEER_TO_WHEEL  # = 2.058 m

# Minimum angle (rad) to consider the vehicle to be turning
STRAIGHT_THRESHOLD = 1e-2

# Joint names exactly as declared in the URDF
STEER_JOINT_NAMES = [
    "front_right_leg",
    "front_left_leg",
    "back_right_leg",
    "back_left",  # NOTE: original URDF uses "back_left", not "back_left_leg"
]
DRIVE_JOINT_NAMES = [
    "front_right_wheel",
    "front_left_wheel",
    "back_right_wheel",
    "back_left_wheel",
]

# Steering angle hard limit (rad) – from URDF joint limits
MAX_STEER_RAD = math.pi


# ---------------------------------------------------------------------------
# 5.  4WS kinematics  +  odometry
# ---------------------------------------------------------------------------
class FourWSKinematics:
    """Forward kinematics : (vx, vy, omega, mode) → (steer_pos[4], drive_vel[4])
    Inverse kinematics : (steer_pos[4], drive_vel[4]) → (vx, vy, omega)

    Array index order matches STEER_JOINT_NAMES / DRIVE_JOINT_NAMES:
      [0] front-right,  [1] front-left,
      [2] back-right,   [3] back-left

    Modes
    -----
    0  Stopped
    1  Opposite-phase  – front/rear steer opposite; standard 4WS cornering.
                         Uses cmd_vel.linear.x + cmd_vel.angular.z.
    2  In-phase (crab) – all wheels steer same direction; lateral translation.
                         Uses cmd_vel.linear.x + cmd_vel.linear.y.
    3  Pivot turn      – wheels on body diagonals; pure spin about centre.
                         Uses cmd_vel.angular.z.
    """

    def __init__(
        self,
        L: float = WHEELBASE,
        wheel_track: float = WHEEL_TRACK,
        d_steer: float = DIST_STEER_TO_WHEEL,
        r: float = WHEEL_RADIUS,
    ) -> None:
        self.L = L
        self.d_steer = d_steer
        self._steering_track = wheel_track - 2.0 * d_steer
        self.r = r

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------
    def compute(
        self, vx: float, vy: float, omega: float, mode: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (steer_pos rad, drive_vel rad/s), both shape (4,)."""
        if mode == 1:
            return self._opposite_phase(vx, omega)
        if mode == 2:
            return self._in_phase(vx, vy)
        if mode == 3:
            return self._pivot_turn(omega)
        return np.zeros(4), np.zeros(4)

    # ------------------------------------------------------------------
    def _compute_irc_radii(
        self, tan_f: float, tan_r: float, left_turn: bool
    ) -> tuple[float, float, float, float, float]:
        """Given effective axle tangents, return (R_body, r_fr, r_fl, r_rr, r_rl).

        Geometry
        --------
        IRC lateral distance from vehicle centreline:
            R_lat = L / (tan_f − tan_r)

        Longitudinal distance from each axle to the IRC:
            front_long = L · |tan_f| / |tan_f − tan_r|
            rear_long  = L · |tan_r| / |tan_f − tan_r|   ← corrected vs C++ bug

        Per-wheel radius:
            r_i = hypot(axle_long, lateral_to_irc ± d_steer)
        """
        denom = tan_f - tan_r  # non-zero (caller guarantees)
        R_lat_abs = abs(self.L / denom)

        front_long = abs(self.L * tan_f / denom)  # longitudinal axle-to-IRC
        rear_long = abs(
            self.L * tan_r / denom
        )  # (corrected from C++ copy-paste bug)

        inner_lat = R_lat_abs - self._steering_track / 2.0
        outer_lat = R_lat_abs + self._steering_track / 2.0

        # Apply per-wheel d_steer correction (inner shrinks, outer grows)
        inner_w = max(inner_lat - self.d_steer, 0.0)
        outer_w = outer_lat + self.d_steer

        # Body turn radius: from vehicle centre (mid-wheelbase) to IRC
        R_body = math.hypot(R_lat_abs, abs(front_long - self.L / 2.0))

        if left_turn:  # FL/RL = inner, FR/RR = outer
            r_fl = math.hypot(front_long, inner_w)
            r_fr = math.hypot(front_long, outer_w)
            r_rl = math.hypot(rear_long, inner_w)
            r_rr = math.hypot(rear_long, outer_w)
        else:  # FR/RR = inner, FL/RL = outer
            r_fr = math.hypot(front_long, inner_w)
            r_fl = math.hypot(front_long, outer_w)
            r_rr = math.hypot(rear_long, inner_w)
            r_rl = math.hypot(rear_long, outer_w)

        return R_body, r_fr, r_fl, r_rr, r_rl

    # ------------------------------------------------------------------
    def _opposite_phase(
        self,
        vx: float,
        omega: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dual-Ackermann: front steers ±δ, rear steers ∓δ."""
        steer = np.zeros(4)
        drive = np.zeros(4)

        if abs(vx) < STRAIGHT_THRESHOLD and abs(omega) < STRAIGHT_THRESHOLD:
            return steer, drive

        # Per-wheel Ackermann steer angles
        denom_r = 2.0 * vx + omega * self._steering_track
        denom_l = 2.0 * vx - omega * self._steering_track
        if abs(denom_r) > STRAIGHT_THRESHOLD:
            steer[0] = math.atan(omega * self.L / denom_r)  # FR
        if abs(denom_l) > STRAIGHT_THRESHOLD:
            steer[1] = math.atan(omega * self.L / denom_l)  # FL
        steer[2] = -steer[0]  # RR (opposite phase)
        steer[3] = -steer[1]  # RL
        steer = np.clip(steer, -MAX_STEER_RAD, MAX_STEER_RAD)

        # Effective axle tangent via harmonic mean of inner/outer wheel tangents
        tan_fr, tan_fl = math.tan(steer[0]), math.tan(steer[1])
        tan_rr, tan_rl = math.tan(steer[2]), math.tan(steer[3])
        s_f = tan_fl + tan_fr
        s_r = tan_rl + tan_rr
        tan_f = 2.0 * tan_fl * tan_fr / s_f if abs(s_f) > 1e-9 else 0.0
        tan_r = 2.0 * tan_rl * tan_rr / s_r if abs(s_r) > 1e-9 else 0.0

        denom = tan_f - tan_r
        if abs(denom) < STRAIGHT_THRESHOLD:
            # Degenerate: drive straight
            speed = vx / self.r
            drive[:] = speed
            return steer, drive

        # IRC-based per-wheel speeds
        left_turn = omega >= 0.0
        R_body, r_fr, r_fl, r_rr, r_rl = self._compute_irc_radii(
            tan_f, tan_r, left_turn
        )

        sign = (
            math.copysign(1.0, vx)
            if abs(vx) > STRAIGHT_THRESHOLD
            else math.copysign(1.0, omega)
        )
        # body_speed: commanded forward speed at vehicle centre
        body_speed = (
            abs(vx) if abs(vx) > STRAIGHT_THRESHOLD else abs(omega) * R_body
        )

        drive[0] = sign * body_speed * r_fr / (R_body * self.r)
        drive[1] = sign * body_speed * r_fl / (R_body * self.r)
        drive[2] = sign * body_speed * r_rr / (R_body * self.r)
        drive[3] = sign * body_speed * r_rl / (R_body * self.r)

        return steer, drive

    # ------------------------------------------------------------------
    def _in_phase(self, vx: float, vy: float) -> tuple[np.ndarray, np.ndarray]:
        """All wheels steer identically – crab / diagonal translation."""
        steer = np.zeros(4)
        drive = np.zeros(4)

        if abs(vx) < STRAIGHT_THRESHOLD and abs(vy) < STRAIGHT_THRESHOLD:
            return steer, drive

        angle = math.atan2(vy, vx)
        angle = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, angle))
        speed = math.hypot(vx, vy) / self.r
        sign = math.copysign(1.0, vx) if abs(vx) > STRAIGHT_THRESHOLD else 1.0

        steer[:] = angle
        drive[:] = sign * speed
        return steer, drive

    # ------------------------------------------------------------------
    def _pivot_turn(self, omega: float) -> tuple[np.ndarray, np.ndarray]:
        """Wheels on body diagonals – zero-radius spin about vehicle centre."""
        steer = np.zeros(4)
        drive = np.zeros(4)

        if abs(omega) < STRAIGHT_THRESHOLD:
            return steer, drive

        half_ang = math.atan2(self.L / 2.0, self._steering_track / 2.0)
        corner_r = math.hypot(self.L / 2.0, self._steering_track / 2.0)
        speed = omega * corner_r / self.r

        steer[0] = -half_ang  # FR
        steer[1] = half_ang  # FL
        steer[2] = half_ang  # RR
        steer[3] = -half_ang  # RL

        drive[0] = -speed
        drive[1] = speed
        drive[2] = -speed
        drive[3] = speed

        return steer, drive

    # ------------------------------------------------------------------
    # Inverse kinematics (odometry)
    # ------------------------------------------------------------------
    def compute_odometry(
        self, steer_pos: np.ndarray, drive_vel: np.ndarray
    ) -> tuple[float, float, float]:
        """Actual joint states → body velocity in body frame (vx, vy, omega).

        steer_pos : [FR, FL, RR, RL] rad   – measured steering joint positions
        drive_vel : [FR, FL, RR, RL] rad/s – measured drive joint velocities

        Algorithm
        ---------
        1. Compute effective axle tangents via harmonic mean of per-wheel tangents
           (matches compute_odometry in the C++ four_wheel_steering_controller).
        2. If turning (|δ_front − δ_rear| > threshold): use IRC geometry to derive
           per-wheel radii and scale wheel speeds to body-centre reference speed.
        3. If straight / in-phase: average wheel speeds, mean steer angle gives
           direction (handles crab-walk case).
        """
        δ_fr, δ_fl, δ_rr, δ_rl = steer_pos
        v_fr = drive_vel[0] * self.r  # wheel ground speed m/s
        v_fl = drive_vel[1] * self.r
        v_rr = drive_vel[2] * self.r
        v_rl = drive_vel[3] * self.r

        front_rear_delta = abs(δ_fl - δ_rl)
        is_turning = (
            abs(δ_fl) > STRAIGHT_THRESHOLD or abs(δ_rl) > STRAIGHT_THRESHOLD
        ) and front_rear_delta > STRAIGHT_THRESHOLD

        if is_turning:
            tan_fl = math.tan(δ_fl)
            tan_fr = math.tan(δ_fr)
            tan_rl = math.tan(δ_rl)
            tan_rr = math.tan(δ_rr)

            s_f = tan_fl + tan_fr
            s_r = tan_rl + tan_rr
            tan_f = 2.0 * tan_fl * tan_fr / s_f if abs(s_f) > 1e-9 else 0.0
            tan_r = 2.0 * tan_rl * tan_rr / s_r if abs(s_r) > 1e-9 else 0.0

            denom = tan_f - tan_r
            if abs(denom) < STRAIGHT_THRESHOLD:
                # In-phase (crab) detected from actual state
                v_body = (v_fr + v_fl + v_rr + v_rl) / 4.0
                mean_angle = (δ_fl + δ_fr + δ_rl + δ_rr) / 4.0
                return (
                    v_body * math.cos(mean_angle),
                    v_body * math.sin(mean_angle),
                    0.0,
                )

            left_turn = δ_fl >= 0.0
            R_body, r_fr, r_fl, r_rr, r_rl = self._compute_irc_radii(
                tan_f, tan_r, left_turn
            )

            # Scale each wheel's speed to the body-centre reference speed, average
            def safe_scale(v: float, r_i: float) -> float:
                return v * R_body / r_i if r_i > 1e-6 else 0.0

            v_body = (
                safe_scale(v_fr, r_fr)
                + safe_scale(v_fl, r_fl)
                + safe_scale(v_rr, r_rr)
                + safe_scale(v_rl, r_rl)
            ) / 4.0

            sign = math.copysign(
                1.0, δ_fl
            )  # positive = left turn → positive omega
            omega = sign * abs(v_body) / R_body
            vx = v_body  # body moves forward (omega handles heading change)
            vy = 0.0

        else:
            # Straight driving or very gentle curve / crab walk
            v_body = (v_fr + v_fl + v_rr + v_rl) / 4.0
            mean_angle = (δ_fl + δ_fr + δ_rl + δ_rr) / 4.0
            vx = v_body * math.cos(mean_angle)
            vy = v_body * math.sin(mean_angle)
            omega = 0.0

        return vx, vy, omega


# ---------------------------------------------------------------------------
# 6.  ROS 2 node (runs in a background daemon thread)
# ---------------------------------------------------------------------------
class FourWSROS2Node(Node):
    """Subscribes /cmd_vel and /four_ws_mode.
    Publishes  /odom  (nav_msgs/Odometry) from Isaac Sim joint feedback.

    Thread model
    ------------
    The node spins in a daemon thread.  The main (Isaac Sim) thread calls
    get_command() and publish_odometry() — both are protected by a lock.
    """

    def __init__(self) -> None:
        super().__init__("four_ws_controller")
        self._lock = threading.Lock()

        # --- command state ---
        self._vx = 0.0
        self._vy = 0.0
        self._omega = 0.0
        self._mode = 0

        # --- odometry pose (integrated in main thread, read here for publish) ---
        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_yaw = 0.0  # rad

        # --- subscribers ---
        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_cb, 10)
        self.create_subscription(Int32, "/four_ws_mode", self._mode_cb, 10)

        # --- publisher ---
        self._odom_pub = self.create_publisher(Odometry, "/odom", 10)

        self.get_logger().info(
            "four_ws_controller ready — /cmd_vel + /four_ws_mode → /odom",
        )

    # ------------------------------------------------------------------
    def _cmd_vel_cb(self, msg: Twist) -> None:
        with self._lock:
            self._vx = msg.linear.x
            self._vy = msg.linear.y
            self._omega = msg.angular.z

    def _mode_cb(self, msg: Int32) -> None:
        with self._lock:
            self._mode = msg.data

    # ------------------------------------------------------------------
    def get_command(self) -> tuple[float, float, float, int]:
        with self._lock:
            return self._vx, self._vy, self._omega, self._mode

    # ------------------------------------------------------------------
    def publish_odometry(
        self,
        x: float,
        y: float,
        yaw: float,
        vx: float,
        vy: float,
        omega: float,
        sim_time_s: float,
    ) -> None:
        """Build and publish a nav_msgs/Odometry message.
        Called from the Isaac Sim main thread — fast, no ROS executor needed.
        """
        msg = Odometry()
        msg.header.frame_id = "odom"
        msg.header.stamp = Time(seconds=sim_time_s).to_msg()
        msg.child_frame_id = "base_link"

        # Pose
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0

        # Quaternion from yaw
        half = yaw * 0.5
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = math.sin(half)
        msg.pose.pose.orientation.w = math.cos(half)

        # Twist (body frame)
        msg.twist.twist.linear.x = vx
        msg.twist.twist.linear.y = vy
        msg.twist.twist.angular.z = omega

        self._odom_pub.publish(msg)


# ---------------------------------------------------------------------------
# 7.  URDF loading helper
# ---------------------------------------------------------------------------
def _patch_and_import_urdf(
    urdf_path: str, dest_prim_path: str = "/World/Tipard"
) -> str:
    """1. Replace ``package://assets/`` → ``file:///absolute/path/assets/``
       so Isaac Sim can find the STL meshes.
    2. Import via omni.kit.commands URDFParseAndImportFile.
    Returns the prim path used.
    """
    assets_dir = os.path.join(
        os.path.dirname(os.path.abspath(urdf_path)), "assets"
    )
    # Normalise to forward slashes for the URI
    assets_uri = assets_dir.replace("\\", "/")

    with open(urdf_path, encoding="utf-8") as fh:
        content = fh.read()

    content = re.sub(r"package://assets/", f"file:///{assets_uri}/", content)

    tmp = tempfile.NamedTemporaryFile(
        suffix=".urdf", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()

    cfg = _urdf.ImportConfig()
    cfg.merge_fixed_joints = False
    cfg.convex_decomp = False
    cfg.import_inertia_tensor = True  # use what the URDF provides
    cfg.fix_base = False  # mobile robot
    cfg.default_drive_strength = 1e5
    cfg.default_position_drive_damping = 1e3
    cfg.default_drive_type = 1  # position drives (we override below)

    success, prim_path_out = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=tmp.name,
        import_config=cfg,
        dest_path=dest_prim_path,
    )

    os.unlink(tmp.name)

    if not success:
        raise RuntimeError(f"URDF import failed: {urdf_path}")

    carb.log_info(f"[4WS] URDF imported → {prim_path_out}")
    return dest_prim_path


# ---------------------------------------------------------------------------
# 8.  Main simulation loop
# ---------------------------------------------------------------------------
def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "tipard", "tipard.urdf")

    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # --- World ---
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # --- Load robot ---
    robot_prim_path = _patch_and_import_urdf(urdf_path)
    world.reset()

    robot = world.scene.add(
        Robot(
            prim_path=robot_prim_path,
            name="tipard",
            position=np.array([0.0, 0.0, 1.5]),  # spawn clear of ground plane
        ),
    )
    world.reset()
    robot.initialize()

    # --- Discover DOF indices ---
    dof_names = list(robot.dof_names)
    print(f"[4WS] DOF names: {dof_names}")

    try:
        steer_idx = np.array(
            [robot.get_dof_index(n) for n in STEER_JOINT_NAMES]
        )
        drive_idx = np.array(
            [robot.get_dof_index(n) for n in DRIVE_JOINT_NAMES]
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve joint indices.  Actual DOF list: {dof_names}",
        ) from exc

    print(
        f"[4WS] Steer indices: {list(zip(STEER_JOINT_NAMES, steer_idx.tolist()))}"
    )
    print(
        f"[4WS] Drive  indices: {list(zip(DRIVE_JOINT_NAMES, drive_idx.tolist()))}"
    )

    # --- Configure drive/steer gains via ArticulationController ---
    #   Steering joints → high position gain, moderate damping
    #   Drive joints    → zero position gain, non-zero velocity damping
    #                     (this makes them behave as velocity-controlled)
    ctrl = robot.get_articulation_controller()

    n_dofs = robot.num_dof
    kps = np.zeros(n_dofs)
    kds = np.zeros(n_dofs)

    for idx in steer_idx:
        kps[idx] = 5e4  # stiff position servo
        kds[idx] = 1e3

    for idx in drive_idx:
        kps[idx] = 0.0  # pure velocity control – no position restoring force
        kds[idx] = 1e2  # damping to track velocity target

    ctrl.set_gains(kps=kps, kds=kds)

    # --- ROS 2 ---
    rclpy.init()
    ros_node = FourWSROS2Node()
    ros_thread = threading.Thread(
        target=rclpy.spin,
        args=(ros_node,),
        daemon=True,
    )
    ros_thread.start()

    # --- Kinematics calculator ---
    kin = FourWSKinematics()

    # --- Odometry pose state (integrated in main thread) ---
    odom_x, odom_y, odom_yaw = 0.0, 0.0, 0.0

    # --- Simulation loop ---
    world.reset()
    dt = world.get_physics_dt()  # fixed physics timestep (s)
    print(f"[4WS] Simulation running (dt={dt:.4f} s).  Ctrl-C to quit.")

    try:
        while simulation_app.is_running():
            world.step(render=not args.headless)

            if not world.is_playing():
                continue

            # --- Forward kinematics: cmd_vel → joint targets ---
            vx, vy, omega, mode = ros_node.get_command()
            steer_pos, drive_vel = kin.compute(vx, vy, omega, mode)

            robot.set_joint_positions(
                positions=steer_pos, joint_indices=steer_idx
            )
            robot.set_joint_velocities(
                velocities=drive_vel, joint_indices=drive_idx
            )

            # --- Inverse kinematics: actual joint states → odometry ---
            all_pos = robot.get_joint_positions()
            all_vel = robot.get_joint_velocities()

            steer_actual = all_pos[steer_idx]
            drive_actual = all_vel[drive_idx]

            vx_o, vy_o, omega_o = kin.compute_odometry(
                steer_actual, drive_actual
            )

            # Integrate pose in odom frame
            odom_x += (
                vx_o * math.cos(odom_yaw) - vy_o * math.sin(odom_yaw)
            ) * dt
            odom_y += (
                vx_o * math.sin(odom_yaw) + vy_o * math.cos(odom_yaw)
            ) * dt
            odom_yaw += omega_o * dt

            sim_time = world.current_time
            ros_node.publish_odometry(
                odom_x, odom_y, odom_yaw, vx_o, vy_o, omega_o, sim_time
            )

    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
