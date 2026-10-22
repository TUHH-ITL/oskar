"""Forward and inverse kinematics for the UR20 on the tipard base.

Constants are the joint origins of tipard_ur20_combined.urdf, so this chain and
the robot the simulator loads cannot drift apart silently; check_against_urdf()
re-reads them and fails if they do.

The scan moves the camera in small steps along a known path, so inverse
kinematics is damped least squares seeded from the current joint angles. That
avoids the branch selection an analytic UR solution needs, and keeps the arm on
the configuration it is already in instead of flipping elbow or wrist mid-scan.
"""

import numpy as np

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# (translation, rpy) of each joint relative to the previous link.
BASE_TO_ARM = ((-0.006, -0.42, 0.881), (1.5708, 1.5708, 0.0))
JOINT_ORIGINS = [
    ((0.0, 0.0, 0.2363), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.570796327, 0.0, 0.0)),
    ((-0.862, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-0.7287, 0.0, 0.201), (0.0, 0.0, 0.0)),
    ((0.0, -0.1593, 0.0), (1.570796327, 0.0, 0.0)),
    ((0.0, 0.1543, 0.0), (1.570796326589793, 3.141592653589793, 3.141592653589793)),
]

# Every UR20 joint turns about its own z.
JOINT_AXIS = np.array([0.0, 0.0, 1.0])

JOINT_LIMITS = np.array([
    [-2 * np.pi, 2 * np.pi],
    [-2 * np.pi, 2 * np.pi],
    [-np.pi, np.pi],
    [-2 * np.pi, 2 * np.pi],
    [-2 * np.pi, 2 * np.pi],
    [-2 * np.pi, 2 * np.pi],
])

MAX_REACH_M = 1.75


def rpy_to_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def transform(translation, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(*rpy)
    T[:3, 3] = translation
    return T


def rotation_z(angle):
    T = np.eye(4)
    c, s = np.cos(angle), np.sin(angle)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    return T


def forward_kinematics(joints, tool=None):
    """Pose of wrist_3_link (or of `tool` mounted on it) in the base_link frame."""
    T = transform(*BASE_TO_ARM)
    for angle, (translation, rpy) in zip(joints, JOINT_ORIGINS):
        T = T @ transform(translation, rpy) @ rotation_z(angle)
    return T if tool is None else T @ tool


def link_poses(joints):
    """Pose of every arm link in the base_link frame, base first."""
    T = transform(*BASE_TO_ARM)
    poses = [T.copy()]
    for angle, (translation, rpy) in zip(joints, JOINT_ORIGINS):
        T = T @ transform(translation, rpy) @ rotation_z(angle)
        poses.append(T.copy())
    return poses


def jacobian(joints, tool=None):
    """Geometric Jacobian of the tool point, in base_link coordinates."""
    poses = link_poses(joints)
    end = poses[-1] if tool is None else poses[-1] @ tool
    p_end = end[:3, 3]
    J = np.zeros((6, len(joints)))
    for i in range(len(joints)):
        # Joint i turns about the z of the frame it produces, poses[i + 1].
        axis = poses[i + 1][:3, :3] @ JOINT_AXIS
        origin = poses[i + 1][:3, 3]
        J[:3, i] = np.cross(axis, p_end - origin)
        J[3:, i] = axis
    return J


def rotation_error(R_current, R_target):
    """Rotation from current to target as a rotation vector in base coordinates."""
    R = R_target @ R_current.T
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2.0 * np.sin(angle)) * angle


def inverse_kinematics(target, seed, tool=None, position_tolerance=1e-3,
                       rotation_tolerance=5e-3, max_iterations=200, damping=0.05,
                       max_step=0.2):
    """Damped least squares IK. Returns (joints, converged, position_error_m)."""
    joints = np.asarray(seed, dtype=float).copy()
    position_error = np.inf
    for _ in range(max_iterations):
        current = forward_kinematics(joints, tool)
        error = np.zeros(6)
        error[:3] = target[:3, 3] - current[:3, 3]
        error[3:] = rotation_error(current[:3, :3], target[:3, :3])
        position_error = float(np.linalg.norm(error[:3]))
        if position_error < position_tolerance and np.linalg.norm(error[3:]) < rotation_tolerance:
            return joints, True, position_error

        J = jacobian(joints, tool)
        # (J J^T + lambda^2 I)^-1 stays finite through singularities, where the
        # arm must pass to reach straight up along a tree.
        step = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), error)
        norm = np.linalg.norm(step)
        if norm > max_step:
            step *= max_step / norm
        joints = np.clip(joints + step, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    return joints, False, position_error


def check_against_urdf(urdf_path):
    """Compare the constants above with the URDF. Returns a list of mismatches."""
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf_path).getroot()
    origins = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        if origin is None:
            continue
        origins[joint.get("name")] = (
            tuple(float(v) for v in origin.get("xyz", "0 0 0").split()),
            tuple(float(v) for v in origin.get("rpy", "0 0 0").split()),
        )

    problems = []
    expected = [("tipard_to_ur20_mount_joint", BASE_TO_ARM)]
    expected += list(zip(JOINT_NAMES, JOINT_ORIGINS))
    for name, (translation, rpy) in expected:
        if name not in origins:
            problems.append(f"{name}: absent from {urdf_path}")
            continue
        urdf_translation, urdf_rpy = origins[name]
        if not np.allclose(urdf_translation, translation, atol=1e-4):
            problems.append(f"{name}: xyz {urdf_translation} in URDF, {translation} here")
        if not np.allclose(urdf_rpy, rpy, atol=1e-4):
            problems.append(f"{name}: rpy {urdf_rpy} in URDF, {rpy} here")
    return problems
