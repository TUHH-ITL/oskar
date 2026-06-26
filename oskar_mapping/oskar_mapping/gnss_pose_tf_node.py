#!/usr/bin/env python3
"""Step 2b: Publish the robot pose as TF from the pre-extracted GNSS trajectory.

Loads pose_trajectory.npz (from extract_pose_trajectory.py). For each camera frame
(it listens to /static_camera_info_1 just to get the frame's timestamp), it interpolates
the robot pose at that instant and broadcasts:

    map -> base_link            (position from RTK ENU, yaw from GNSS course-over-ground)
    base_link -> <camera_frame> (STATIC mounting; cameras look sideways at the trees)

This lets backprojection_node look up map->camera at each detection's timestamp.

The camera mounting is a first GUESS (parameterised) — refine it by checking that the
back-projected flowers land on the geojson tree points.

Run inside the venv with the feeder publishing /static_camera_info_1:
    python3 gnss_pose_tf_node.py
"""

import math
import os

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

DEFAULT_NPZ = os.path.expanduser(
    "~/ros2_ws/src/oskar/oskar_mapping/bagfile_data/pose_trajectory.npz"
)


def mat_to_quat(R):
    """3x3 rotation matrix -> (x,y,z,w)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


def rpy_to_mat(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


class GnssPoseTfNode(Node):
    def __init__(self):
        super().__init__("gnss_pose_tf_node")

        self.declare_parameter("trajectory_file", DEFAULT_NPZ)
        self.declare_parameter("stamp_topic", "/static_camera_info_1")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "static_camera_1")
        # camera mounting (base_link -> camera optical). First guess; tune later.
        self.declare_parameter("camera_side", "right")          # "right" or "left" of travel
        self.declare_parameter("camera_xyz", [0.1, 0.0, 1.0])   # mount position in base_link (m)
        self.declare_parameter("camera_tilt_down_deg", 0.0)     # extra downward tilt
        self.declare_parameter("camera_yaw_offset_deg", 0.0)    # extra yaw tweak (azimuth)

        npz = self.get_parameter("trajectory_file").value
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value

        d = np.load(npz)
        self.t = d["t"]
        self.E, self.N, self.U, self.yaw = d["E"], d["N"], d["U"], d["yaw"]
        # unwrap yaw so linear interpolation doesn't break at the ±pi seam
        self._cos = np.cos(self.yaw)
        self._sin = np.sin(self.yaw)
        self.get_logger().info(
            f"Loaded {len(self.t)} poses, t=[{self.t[0]:.1f},{self.t[-1]:.1f}]"
        )

        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        self._publish_camera_mount()

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(
            CameraInfo, self.get_parameter("stamp_topic").value, self._on_stamp, qos
        )
        self.get_logger().info("gnss_pose_tf_node ready — publishing map->base_link per frame")

    # ------------------------------------------------------------------
    def _publish_camera_mount(self):
        side = self.get_parameter("camera_side").value
        xyz = self.get_parameter("camera_xyz").value
        tilt = math.radians(self.get_parameter("camera_tilt_down_deg").value)
        yawoff = math.radians(self.get_parameter("camera_yaw_offset_deg").value)

        # base_link: x fwd, y left, z up.  camera optical: x right, y down, z forward(out of lens).
        # Camera looks sideways at the row: optical z -> base -y (right) or +y (left).
        if side == "left":
            # optical_x=+x, optical_y=-z, optical_z=+y
            R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
        else:  # right
            # optical_x=-x, optical_y=-z, optical_z=-y
            R = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=float)
        # apply small tuning offsets (downward tilt about optical x, azimuth yaw about base z)
        R = rpy_to_mat(0.0, tilt, yawoff) @ R
        qx, qy, qz, qw = mat_to_quat(R)

        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.base_frame
        tf.child_frame_id = self.camera_frame
        tf.transform.translation.x = float(xyz[0])
        tf.transform.translation.y = float(xyz[1])
        tf.transform.translation.z = float(xyz[2])
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.static_tf.sendTransform(tf)
        self.get_logger().info(f"Static base_link->{self.camera_frame} published (side={side})")

    # ------------------------------------------------------------------
    def _on_stamp(self, msg: CameraInfo):
        st = msg.header.stamp
        tq = st.sec + st.nanosec / 1e9

        e = float(np.interp(tq, self.t, self.E))
        n = float(np.interp(tq, self.t, self.N))
        u = float(np.interp(tq, self.t, self.U))
        c = float(np.interp(tq, self.t, self._cos))
        s = float(np.interp(tq, self.t, self._sin))
        yaw = math.atan2(s, c)

        tf = TransformStamped()
        tf.header.stamp = st
        tf.header.frame_id = self.map_frame
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x = e
        tf.transform.translation.y = n
        tf.transform.translation.z = u
        tf.transform.rotation.z = math.sin(yaw / 2.0)
        tf.transform.rotation.w = math.cos(yaw / 2.0)
        self.tf.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = GnssPoseTfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
