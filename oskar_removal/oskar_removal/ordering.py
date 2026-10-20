"""Order in which the arm visits the targets."""

import numpy as np


def order_nearest_neighbour(positions, start=None):
    """Visiting order over positions, always moving to the nearest unvisited one.

    Starts from the position nearest `start`, or from the highest one when no
    start is given.
    """
    points = np.asarray(positions, dtype=float).reshape(-1, 3)
    if not len(points):
        return []
    if start is None:
        current = int(np.argmax(points[:, 2]))
    else:
        current = int(np.argmin(np.linalg.norm(points - np.asarray(start, float), axis=1)))
    order = [current]
    left = set(range(len(points))) - {current}
    while left:
        rest = np.array(sorted(left))
        current = int(rest[np.argmin(np.linalg.norm(points[rest] - points[current], axis=1))])
        order.append(current)
        left.discard(current)
    return order
