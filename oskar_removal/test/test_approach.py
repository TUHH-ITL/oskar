import numpy as np
import pytest

from oskar_removal.approach import (approach_axis, approach_pose, frame_with_z,
                                     matrix_from_quaternion, quaternion_from_matrix,
                                     tip_to_link)


@pytest.mark.parametrize("axis", [(0, -1, 0), (0, 0, 1), (1, 1, 0), (0.2, -0.9, 0.3)])
def test_frame_is_a_rotation_with_the_requested_z(axis):
    R = frame_with_z(axis)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R[:, 2], np.array(axis) / np.linalg.norm(axis))


def test_quaternion_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        R = frame_with_z(rng.normal(size=3), rng.normal(size=3))
        assert np.allclose(matrix_from_quaternion(quaternion_from_matrix(R)), R, atol=1e-9)


def test_stand_off_sits_on_the_axis_and_looks_back():
    target = np.array([1.0, 2.0, 1.5])
    axis = np.array([0.0, -1.0, 0.0])
    tip, q = approach_pose(target, axis, 0.1)
    assert np.allclose(tip, [1.0, 1.9, 1.5])
    R = matrix_from_quaternion(q)
    assert np.allclose(R[:, 2], -axis)
    # Roll from the reference: tool +X points up for a horizontal approach.
    assert np.allclose(R[:, 0], [0, 0, 1])


def test_tip_to_link_moves_back_along_tool_z():
    tip, q = approach_pose([0, 0, 0], [0, -1, 0], 0.1)
    link, q_link = tip_to_link(tip, q, 0.13)
    assert np.allclose(link, [0, -0.23, 0])
    assert np.allclose(q_link, q)


def test_axis_falls_back_to_away_from_the_trunk():
    assert np.allclose(approach_axis([1, 0, 2], [0, 0, 0], trunk_xy=(0, 0)), [1, 0, 0])
    assert np.allclose(approach_axis([1, 0, 2], [0, 0, 3]), [0, 0, 1])
    with pytest.raises(ValueError):
        approach_axis([0, 0, 2], [0, 0, 0], trunk_xy=(0, 0))
