import numpy as np

from oskar_removal.ordering import order_nearest_neighbour


def test_nearest_neighbour_order_visits_all_once():
    points = np.random.default_rng(4).uniform(size=(30, 3))
    order = order_nearest_neighbour(points, start=[0, 0, 0])
    assert sorted(order) == list(range(30))
    assert order[0] == int(np.argmin(np.linalg.norm(points, axis=1)))
    assert order_nearest_neighbour(points)[0] == int(np.argmax(points[:, 2]))
