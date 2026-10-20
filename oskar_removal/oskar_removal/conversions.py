"""Between the landmark map and the thinning interfaces."""

import math

import numpy as np
from geometry_msgs.msg import Pose, PoseStamped

from oskar_msgs.msg import ThinningCandidate, ThinningCandidates
from oskar_removal.approach import approach_axis, frame_with_z, quaternion_from_matrix


def candidates_from_landmarks(landmarks, tree_id, trunk_xy, header):
    """ThinningCandidates for one tree from its Landmark list.

    A landmark with neither a view direction nor a usable trunk offset has no
    approach axis and is left out; the count of those is returned alongside.
    """
    array = ThinningCandidates()
    array.header = header
    array.tree_id = int(tree_id)
    skipped = 0
    for lm in landmarks:
        position = np.array([lm.position.x, lm.position.y, lm.position.z])
        view = np.array([lm.view_direction.x, lm.view_direction.y, lm.view_direction.z])
        try:
            axis = approach_axis(position, view, trunk_xy)
        except ValueError:
            skipped += 1
            continue
        c = ThinningCandidate()
        c.id = int(lm.landmark_id)
        c.tree_id = int(tree_id)
        c.class_label = lm.class_label
        c.pose.header = header
        c.pose.pose = pose_from(position, quaternion_from_matrix(frame_with_z(axis)))
        c.diameter = float(lm.diameter) if lm.diameter > 0 else math.nan
        c.confidence = float(lm.mean_confidence)
        c.observation_count = int(lm.observation_count)
        array.candidates.append(c)
    return array, skipped


def pose_from(position, quaternion):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in position)
    (pose.orientation.x, pose.orientation.y,
     pose.orientation.z, pose.orientation.w) = (float(v) for v in quaternion)
    return pose


def stamped(pose, header):
    msg = PoseStamped()
    msg.header = header
    msg.pose = pose
    return msg


def position_of(pose):
    return np.array([pose.position.x, pose.position.y, pose.position.z])


def quaternion_of(pose):
    o = pose.orientation
    return np.array([o.x, o.y, o.z, o.w])
