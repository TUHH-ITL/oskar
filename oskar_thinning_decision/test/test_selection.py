from itertools import combinations

import numpy as np
import pytest

from oskar_thinning_decision.selection import min_spacing, select_keep


def test_keeps_everything_under_the_limit():
    points = np.random.default_rng(0).uniform(size=(5, 3))
    assert select_keep(points, 5) == [0, 1, 2, 3, 4]
    assert select_keep(points, 9) == [0, 1, 2, 3, 4]


def test_respects_the_limit_and_keeps_forced():
    points = np.random.default_rng(1).uniform(size=(60, 3))
    keep = select_keep(points, 12, forced=[3, 7])
    assert len(keep) == 12
    assert {3, 7} <= set(keep)


def test_forced_beyond_the_limit_keeps_only_forced():
    points = np.random.default_rng(2).uniform(size=(20, 3))
    assert select_keep(points, 2, forced=[1, 4, 9]) == [1, 4, 9]


@pytest.mark.parametrize("seed", range(8))
def test_close_to_the_best_closest_pair(seed):
    points = np.random.default_rng(seed).uniform(size=(13, 3))
    limit = 5
    best = max(min_spacing(points, c) for c in combinations(range(len(points)), limit))
    got = min_spacing(points, select_keep(points, limit))
    # Farthest-point insertion guarantees half the optimum; the swaps do better.
    assert got >= 0.5 * best
    assert got >= 0.8 * best


def test_beats_random_subsets():
    rng = np.random.default_rng(3)
    points = rng.uniform(size=(150, 3))
    got = min_spacing(points, select_keep(points, 20))
    random = [min_spacing(points, rng.choice(150, 20, replace=False)) for _ in range(200)]
    assert got > max(random)


def test_drops_one_of_two_neighbours():
    points = np.array([[0, 0, 0], [1, 0, 0], [1.01, 0, 0], [2, 0, 0]], dtype=float)
    keep = select_keep(points, 3)
    assert 0 in keep and 3 in keep
    assert len({1, 2} & set(keep)) == 1

