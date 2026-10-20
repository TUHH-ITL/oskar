"""Approach geometry: candidate frames, tool stand-off poses, tool tip to flange.

Conventions, from the interface contract:
  Candidate +Z    approach axis, out of the canopy into free space; roll is meaningless.
  Tool tip +Z     points at the object, so it is the candidate +Z reversed.
  Tool roll       not inherited from the candidate; tool +X is the reference
                  direction (world up by default) made perpendicular to +Z.
"""

import numpy as np


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def frame_with_z(z_axis, reference=(0.0, 0.0, 1.0)):
    """Rotation matrix whose +Z is z_axis and whose +X leans toward reference."""
    z = unit(z_axis)
    if not np.any(z):
        raise ValueError("zero axis")
    x = np.asarray(reference, dtype=float) - np.dot(reference, z) * z
    if np.linalg.norm(x) < 1e-6:
        # Reference parallel to the axis: any perpendicular will do.
        x = np.array([1.0, 0.0, 0.0]) - z[0] * z
        if np.linalg.norm(x) < 1e-6:
            x = np.array([0.0, 1.0, 0.0]) - z[1] * z
    x = unit(x)
    return np.column_stack([x, np.cross(z, x), z])


def quaternion_from_matrix(R):
    """(x, y, z, w) of a rotation matrix."""
    R = np.asarray(R, dtype=float)
    trace = np.trace(R)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def matrix_from_quaternion(q):
    x, y, z, w = unit(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def approach_axis(position, view_direction, trunk_xy=None):
    """ThinningCandidate +Z: where the camera saw it from, else horizontally away from the trunk."""
    axis = unit(view_direction)
    if np.any(axis):
        return axis
    if trunk_xy is not None:
        outward = np.array([position[0] - trunk_xy[0], position[1] - trunk_xy[1], 0.0])
        if np.linalg.norm(outward) > 1e-6:
            return unit(outward)
    raise ValueError("no approach axis: no view direction and no trunk position")


def approach_pose(position, axis, standoff, roll_reference=(0.0, 0.0, 1.0)):
    """Tool tip (position, quaternion) at `standoff` along the axis, looking back at the object."""
    axis = unit(axis)
    tip = np.asarray(position, dtype=float) + axis * float(standoff)
    return tip, quaternion_from_matrix(frame_with_z(-axis, roll_reference))


def tip_to_link(position, quaternion, tcp_offset):
    """Pose of the link the tool tip sits `tcp_offset` metres in front of, along link +Z."""
    R = matrix_from_quaternion(quaternion)
    return np.asarray(position, dtype=float) - R[:, 2] * float(tcp_offset), unit(quaternion)
