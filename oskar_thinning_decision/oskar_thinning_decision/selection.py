"""Choose which candidates stay on a tree.

At most `limit` candidates stay, spread as far apart as possible. The kept set is
grown by farthest-point insertion, then refined by swaps that widen its closest
pair. Farthest-point insertion is within a factor of two of the best possible
closest-pair distance, and of the best possible coverage (the largest distance
from any removed candidate to the nearest kept one), so the kept candidates are
both far apart and spread over the whole canopy.

Candidates the robot cannot remove are passed as forced: they stay whatever the
plan says, so the others are chosen around them.
"""

import numpy as np
from scipy.spatial.distance import cdist


def min_spacing(positions, indices):
    """Distance of the closest pair among positions[indices]; NaN for fewer than two."""
    indices = list(indices)
    if len(indices) < 2:
        return float("nan")
    points = np.asarray(positions, dtype=float)[indices]
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


def select_keep(positions, limit, forced=(), max_swaps=None):
    """Indices of the candidates to keep, sorted.

    Returns every index when there are no more than `limit` candidates, and only
    the forced ones when those alone reach the limit.
    """
    points = np.asarray(positions, dtype=float).reshape(-1, 3)
    n = len(points)
    limit = max(0, int(limit))
    forced = sorted({int(i) for i in forced if 0 <= int(i) < n})
    if n <= limit:
        return list(range(n))
    if len(forced) >= limit:
        return forced

    d = cdist(points, points)
    keep = list(forced)
    if not keep:
        if limit == 1:
            centroid = points.mean(axis=0)
            return [int(np.argmin(np.linalg.norm(points - centroid, axis=1)))]
        i, j = np.unravel_index(np.argmax(d), d.shape)
        keep = [int(i), int(j)]

    nearest = d[:, keep].min(axis=1)
    while len(keep) < limit:
        nearest[keep] = -np.inf
        best = int(np.argmax(nearest))
        keep.append(best)
        nearest = np.minimum(nearest, d[:, best])

    return sorted(_widen_closest_pair(d, keep, set(forced), max_swaps or 10 * limit))


def _widen_closest_pair(d, keep, forced, max_swaps):
    """Swap one of the closest kept pair for an unkept candidate while that helps."""
    keep = list(keep)
    n = len(d)
    for _ in range(max_swaps):
        sub = d[np.ix_(keep, keep)].copy()
        np.fill_diagonal(sub, np.inf)
        current = sub.min()
        a, b = np.unravel_index(np.argmin(sub), sub.shape)

        best_gain, best_swap = 0.0, None
        for slot in (a, b):
            if keep[slot] in forced:
                continue
            rest = keep[:slot] + keep[slot + 1:]
            rest_sub = d[np.ix_(rest, rest)].copy()
            np.fill_diagonal(rest_sub, np.inf)
            rest_min = rest_sub.min() if len(rest) > 1 else np.inf
            outside = np.setdiff1d(np.arange(n), keep)
            if not len(outside):
                continue
            reach = d[np.ix_(outside, rest)].min(axis=1)
            k = int(np.argmax(reach))
            gain = min(rest_min, reach[k]) - current
            if gain > best_gain + 1e-12:
                best_gain, best_swap = gain, (slot, int(outside[k]))
        if best_swap is None:
            break
        keep[best_swap[0]] = best_swap[1]
    return keep

