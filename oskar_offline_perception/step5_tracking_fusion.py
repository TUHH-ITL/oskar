#!/usr/bin/env python3
"""Alternative Step 5: merge repeat views of one flower by tracking it across frames.

`step5_multiview_fusion` runs DBSCAN over all world observations at once. Density clustering
asks "are these points close together?", which cannot separate one flower seen twenty times
from four flowers three centimetres apart, and links transitively: any two points within eps
join the same cluster, so at high detection density neighbouring blossoms chain into runaway
clusters. On the MaskDINO run one cluster absorbed 13,421 of 82,917 observations.

Tracking asks the question we actually mean - "is this the same flower?" - and answers it
temporally. The camera advances about 0.2 m per frame at 5 Hz and the pose is known to a few
centimetres, so a flower's world position should barely move between consecutive frames.
Associating one detection per track per frame, and only where the match is mutually nearest,
removes chaining by construction: a track can gain at most one observation per frame, so its
length is bounded by how many frames the flower was visible (tens), not by how many blossoms
happen to be nearby (thousands).

Track length also becomes a free quality measure. A blossom seen in eighteen of the twenty
frames that should have seen it is real; one seen twice is probably a false positive.

Output matches process_multiview_fusion: [x, y, z, confidence, observation_count].
"""

import numpy as np
from scipy.spatial import cKDTree

import config


def _mutual_nearest(track_pos, det_pos, radius):
    """Pairs that are each other's nearest neighbour within radius.

    Requiring the match to be mutual is what prevents a single dense clump of detections
    from being swallowed by one track, and prevents one detection from extending several
    tracks at once.
    """
    if len(track_pos) == 0 or len(det_pos) == 0:
        return []
    det_tree = cKDTree(det_pos)
    track_tree = cKDTree(track_pos)
    d_td, i_td = det_tree.query(track_pos, k=1, distance_upper_bound=radius)
    d_dt, i_dt = track_tree.query(det_pos, k=1, distance_upper_bound=radius)

    pairs = []
    for ti, (dist, di) in enumerate(zip(d_td, i_td)):
        if not np.isfinite(dist) or di >= len(det_pos):
            continue
        if i_dt[di] == ti:                       # mutual
            pairs.append((ti, di, float(dist)))
    return pairs


def process_tracking_fusion(world_observations, match_radius=None, max_gap=None,
                            min_observations=None):
    """Fuse observations into flower landmarks by temporal association.

    world_observations columns: [x, y, z, confidence, instance_id, frame_index].
    The frame index is required - without it there is no temporal ordering to track along,
    and the caller should use process_multiview_fusion instead.
    """
    if match_radius is None:
        match_radius = getattr(config, "TRACK_MATCH_RADIUS_M", 0.06)
    if max_gap is None:
        max_gap = getattr(config, "TRACK_MAX_GAP_FRAMES", 2)
    if min_observations is None:
        min_observations = getattr(config, "TRACK_MIN_OBSERVATIONS", 3)

    obs = np.asarray(world_observations, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[0] == 0:
        return np.array([], dtype=np.float32)
    if obs.shape[1] < 6:
        raise ValueError(
            "tracking fusion needs a frame index in column 5; got "
            f"{obs.shape[1]} columns. Use process_multiview_fusion for observations "
            "without frame information."
        )

    order = np.argsort(obs[:, 5], kind="stable")
    obs = obs[order]
    frames = obs[:, 5]
    unique_frames = np.unique(frames)

    # a track holds running sums so the representative position is the mean of its views
    active = []          # dicts: sum(3), conf_sum, n, last_frame, last_pos
    finished = []

    for f in unique_frames:
        sel = obs[frames == f]
        det_pos = sel[:, :3]
        det_conf = sel[:, 3]

        alive = [t for t in active if f - t["last_frame"] <= max_gap]
        finished.extend(t for t in active if f - t["last_frame"] > max_gap)
        active = alive

        track_pos = np.array([t["last_pos"] for t in active]) if active else np.empty((0, 3))
        pairs = _mutual_nearest(track_pos, det_pos, match_radius)

        matched_tracks = set()
        matched_dets = set()
        for ti, di, _ in sorted(pairs, key=lambda p: p[2]):
            if ti in matched_tracks or di in matched_dets:
                continue
            t = active[ti]
            t["sum"] += det_pos[di]
            t["conf_sum"] += det_conf[di]
            t["n"] += 1
            t["last_pos"] = det_pos[di]
            t["last_frame"] = f
            matched_tracks.add(ti)
            matched_dets.add(di)

        for di in range(len(det_pos)):
            if di in matched_dets:
                continue
            active.append({"sum": det_pos[di].copy(), "conf_sum": float(det_conf[di]),
                           "n": 1, "last_pos": det_pos[di], "last_frame": f})

    finished.extend(active)

    landmarks = [[*(t["sum"] / t["n"]), t["conf_sum"] / t["n"], t["n"]]
                 for t in finished if t["n"] >= min_observations]
    if not landmarks:
        return np.array([], dtype=np.float32)
    return np.array(landmarks, dtype=np.float32)


if __name__ == "__main__":
    import os

    obs_path = os.path.join(config.BACKPROJ_OUT_DIR, "world_obs.npy")
    if not os.path.exists(obs_path):
        print(f"Error: Step 4 output {obs_path} not found. Run previous steps first!")
        raise SystemExit(1)

    world_obs = np.load(obs_path)
    landmarks = process_tracking_fusion(world_obs)
    print(f"Tracked {len(world_obs)} observations into {len(landmarks)} flower landmarks.")
    if len(landmarks):
        n = landmarks[:, 4]
        print(f"  observations per landmark: median {np.median(n):.0f}, max {n.max():.0f}")
