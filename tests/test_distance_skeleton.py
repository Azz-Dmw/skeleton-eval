"""距离变换骨架的单元测试。"""
import pytest

from src.dataset import make_comb, make_rectangle, make_ring
from src.distance_skeleton import distance_skeleton
from src.metrics import skeleton_metrics


def test_distance_skeleton_rectangle():
    binary = make_rectangle()
    m = skeleton_metrics(binary, distance_skeleton(binary))

    assert m["skeleton_pixels"] == 80
    assert m["endpoints"] == 2
    assert m["components"] == 1


def test_distance_skeleton_comb_topology():
    binary = make_comb()
    m = skeleton_metrics(binary, distance_skeleton(binary))

    assert m["endpoints"] == 7
    assert m["branch_points"] == 5
    assert m["components"] == 1


def test_distance_skeleton_ring_hole():
    binary = make_ring()
    m = skeleton_metrics(binary, distance_skeleton(binary))

    assert m["skeleton_holes"] == 1
    assert m["components"] == 1


def test_distance_skeleton_invalid_sigma():
    with pytest.raises(ValueError):
        distance_skeleton(make_rectangle(), sigma=0)