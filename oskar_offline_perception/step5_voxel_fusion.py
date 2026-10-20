#!/usr/bin/env python3
"""Alternative Step 5: deduplicate observations by voxel occupancy, not by association.

Per-flower correspondence is not recoverable in this data: a detection's nearest neighbour
in the previous frame is ~0.104 m away, its nearest neighbour within its own frame ~0.032 m.
Voxelising needs no correspondence - repeat views of one blossom share a voxel.

Voxel size is set by physics, not tuned: above the ~0.05 m observation scatter, near the
~0.06 m corymb diameter, so one voxel is roughly one cluster (the unit of the hand counts).

Output matches process_multiview_fusion: [x, y, z, confidence, observation_count].
"""

import numpy as np

import config


def process_voxel_fusion(world_observations, voxel_size=None, min_observations=None,
                         return_grid=False):
    """Collapse observations onto a voxel grid; each occupied voxel is one flower cluster.

    world_observations columns: [x, y, z, confidence, ...]. Frame index is not required.
    A voxel must be seen at least `min_observations` times to count, which suppresses
    one-off false positives without needing to identify anything.
    """
    if voxel_size is None:
        voxel_size = getattr(config, "VOXEL_SIZE_M", 0.05)
    if min_observations is None:
        min_observations = getattr(config, "VOXEL_MIN_OBSERVATIONS", 3)

    obs = np.asarray(world_observations, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[0] == 0:
        return (np.array([], dtype=np.float32), None) if return_grid else np.array([], dtype=np.float32)

    xyz = obs[:, :3]
    conf = obs[:, 3] if obs.shape[1] > 3 else np.ones(len(obs))

    keys = np.floor(xyz / voxel_size).astype(np.int64)
    uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)

    # centroid and mean confidence of the observations in each voxel
    sums = np.zeros((len(uniq), 3))
    np.add.at(sums, inverse, xyz)
    conf_sums = np.zeros(len(uniq))
    np.add.at(conf_sums, inverse, conf)

    keep = counts >= min_observations
    landmarks = np.column_stack([sums[keep] / counts[keep, None],
                                 conf_sums[keep] / counts[keep],
                                 counts[keep]]).astype(np.float32)
    if return_grid:
        return landmarks, {"keys": uniq[keep], "counts": counts[keep], "voxel_size": voxel_size}
    return landmarks


def distribution_about_tree(landmarks, tree_xy, row_axis, radius=0.7):
    """Where the blossom sits on one tree, in tree-local coordinates.

    Returns along-row, across-row and height offsets relative to the trunk position, which
    is what a per-tree distribution plot needs.
    """
    if len(landmarks) == 0:
        return np.empty((0, 3))
    d = np.linalg.norm(landmarks[:, :2] - np.asarray(tree_xy), axis=1)
    sel = landmarks[d < radius]
    if len(sel) == 0:
        return np.empty((0, 3))
    ax = np.asarray(row_axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    perp = np.array([-ax[1], ax[0]])
    rel = sel[:, :2] - np.asarray(tree_xy)
    return np.column_stack([rel @ ax, rel @ perp, sel[:, 2]])


if __name__ == "__main__":
    import os

    obs_path = os.path.join(config.BACKPROJ_OUT_DIR, "world_obs.npy")
    if not os.path.exists(obs_path):
        print(f"Error: Step 4 output {obs_path} not found. Run previous steps first!")
        raise SystemExit(1)
    world_obs = np.load(obs_path)
    lm = process_voxel_fusion(world_obs)
    print(f"{len(world_obs)} observations -> {len(lm)} occupied voxels "
          f"at {getattr(config,'VOXEL_SIZE_M',0.05)} m")
    if len(lm):
        print(f"  observations per voxel: median {np.median(lm[:,4]):.0f}, max {lm[:,4].max():.0f}")
