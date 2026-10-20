"""Streaming fusion of repeated flower observations into a persistent map.

Two stages, because one is not enough here.

Stage one associates detections across consecutive frames by mutual nearest
neighbour, the online form of the offline pipeline's tracking fusion. It answers
"is this the same flower as last frame?" and, because a track can absorb at most
one detection per frame, cannot chain neighbouring blossoms together the way
density clustering does.

Stage two is what the offline pipeline lacked. A track dies once the flower
leaves view for a few frames, so a flower seen again later starts a fresh track
and is counted twice. Offline that showed up as zero of 2548 landmarks spanning
the two driving passes. A scan revisits the same flowers by design, sweeping up
and then down, so retired tracks are merged into a persistent landmark map by
position instead of being emitted directly. Merging operates on track centroids,
which stage one has already de-duplicated, so it cannot chain.
"""

import numpy as np
from scipy.spatial import cKDTree


class FlowerTracker:
    """Fuse per-frame 3D detections into persistent flower landmarks."""

    def __init__(self, match_radius=0.06, max_gap_frames=2, min_observations=3,
                 merge_radius=0.05):
        self.match_radius = match_radius
        self.max_gap_frames = max_gap_frames
        self.min_observations = min_observations
        self.merge_radius = merge_radius

        self._active = []
        self._landmarks = []
        self._frame = 0
        self._next_landmark_id = 0

    def reset(self):
        self._active = []
        self._landmarks = []
        self._frame = 0

    @property
    def frame_index(self):
        return self._frame

    def update(self, positions, confidences, view_directions=None, diameters=None, labels=None):
        """Add one frame of detections. Positions are (N, 3) in the map frame.

        view_directions are unit vectors toward the camera, diameters in metres;
        both optional and only carried along, never used for association.
        """
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
        views = (np.zeros_like(positions) if view_directions is None
                 else np.asarray(view_directions, dtype=np.float64).reshape(-1, 3))
        sizes = (np.zeros(len(positions)) if diameters is None
                 else np.asarray(diameters, dtype=np.float64).reshape(-1))
        labels = [""] * len(positions) if labels is None else list(labels)
        self._frame += 1
        f = self._frame

        self._retire(lambda t: f - t["last_frame"] > self.max_gap_frames)

        if len(positions):
            track_pos = (np.array([t["last_pos"] for t in self._active])
                         if self._active else np.empty((0, 3)))
            pairs = _mutual_nearest(track_pos, positions, self.match_radius)

            matched_tracks, matched_dets = set(), set()
            for ti, di, _ in sorted(pairs, key=lambda p: p[2]):
                if ti in matched_tracks or di in matched_dets:
                    continue
                t = self._active[ti]
                t["sum"] += positions[di]
                t["conf_sum"] += confidences[di]
                t["view_sum"] += views[di]
                t["diameters"].append(float(sizes[di]))
                t["labels"].append(labels[di])
                t["n"] += 1
                t["last_pos"] = positions[di]
                t["last_frame"] = f
                matched_tracks.add(ti)
                matched_dets.add(di)

            for di in range(len(positions)):
                if di not in matched_dets:
                    self._active.append({
                        "sum": positions[di].copy(),
                        "conf_sum": float(confidences[di]),
                        "view_sum": views[di].copy(),
                        "diameters": [float(sizes[di])],
                        "labels": [labels[di]],
                        "n": 1,
                        "last_pos": positions[di],
                        "last_frame": f,
                    })

    def flush(self):
        """Retire every open track. Call when a scan ends."""
        self._retire(lambda t: True)

    def landmarks(self):
        """(M, 5) array of x, y, z, mean confidence, observation count."""
        if not self._landmarks:
            return np.empty((0, 5))
        return np.array([
            [lm["sum"][0] / lm["n"], lm["sum"][1] / lm["n"], lm["sum"][2] / lm["n"],
             lm["conf_sum"] / lm["n"], lm["n"]]
            for lm in self._landmarks
        ])

    def landmark_ids(self):
        return [lm["id"] for lm in self._landmarks]

    def landmark_geometry(self):
        """(M, 3) unit mean view directions (zero if unknown) and (M,) median diameters."""
        if not self._landmarks:
            return np.empty((0, 3)), np.empty(0)
        views = np.array([lm["view_sum"] for lm in self._landmarks])
        norms = np.linalg.norm(views, axis=1, keepdims=True)
        views = np.divide(views, norms, out=np.zeros_like(views), where=norms > 1e-9)
        diameters = np.array([float(np.median(lm["diameters"])) for lm in self._landmarks])
        return views, diameters

    def landmark_labels(self):
        """Most frequent class label per landmark."""
        return [max(set(lm["labels"]), key=lm["labels"].count) if lm["labels"] else ""
                for lm in self._landmarks]

    @property
    def active_track_count(self):
        return len(self._active)

    @property
    def merge_count(self):
        """Times a retired track joined an existing landmark instead of adding one.

        This is the double count that would otherwise be published.
        """
        return sum(lm["merges"] for lm in self._landmarks)

    def _retire(self, predicate):
        keep = []
        for t in self._active:
            if predicate(t):
                self._absorb(t)
            else:
                keep.append(t)
        self._active = keep

    def _absorb(self, track):
        if track["n"] < self.min_observations:
            return
        centroid = track["sum"] / track["n"]

        if self._landmarks:
            centres = np.array([lm["sum"] / lm["n"] for lm in self._landmarks])
            distances = np.linalg.norm(centres - centroid, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= self.merge_radius:
                lm = self._landmarks[nearest]
                lm["sum"] += track["sum"]
                lm["conf_sum"] += track["conf_sum"]
                lm["view_sum"] += track["view_sum"]
                lm["diameters"].extend(track["diameters"])
                lm["labels"].extend(track["labels"])
                lm["n"] += track["n"]
                lm["merges"] += 1
                return

        self._landmarks.append({
            "id": self._next_landmark_id,
            "sum": track["sum"].copy(),
            "conf_sum": track["conf_sum"],
            "view_sum": track["view_sum"].copy(),
            "diameters": list(track["diameters"]),
            "labels": list(track["labels"]),
            "n": track["n"],
            "merges": 0,
        })
        self._next_landmark_id += 1


def _mutual_nearest(track_pos, det_pos, radius):
    """Pairs that are each other's nearest neighbour within radius.

    Mutuality is what stops one dense clump of detections being swallowed by a
    single track, and one detection from extending several tracks at once.
    """
    if len(track_pos) == 0 or len(det_pos) == 0:
        return []
    det_tree = cKDTree(det_pos)
    track_tree = cKDTree(track_pos)
    _, i_dt = track_tree.query(det_pos, k=1, distance_upper_bound=radius)
    d_td, i_td = det_tree.query(track_pos, k=1, distance_upper_bound=radius)

    pairs = []
    for ti, (dist, di) in enumerate(zip(d_td, i_td)):
        if not np.isfinite(dist) or di >= len(det_pos):
            continue
        if i_dt[di] == ti:
            pairs.append((ti, di, float(dist)))
    return pairs


def assign_to_trees(positions, trees, max_radius):
    """Nearest trunk in XY. Returns an array of tree ids, -1 where none is close enough."""
    positions = np.asarray(positions).reshape(-1, 3)
    if len(positions) == 0 or not trees:
        return np.full(len(positions), -1, dtype=int)
    ids = np.array(sorted(trees))
    centres = np.array([trees[i] for i in ids])
    d = np.linalg.norm(positions[:, None, :2] - centres[None, :, :], axis=2)
    nearest = np.argmin(d, axis=1)
    out = ids[nearest]
    out[d[np.arange(len(positions)), nearest] > max_radius] = -1
    return out
