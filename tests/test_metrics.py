"""骨架评估指标的单元测试。"""
import numpy as np

from src.dataset import make_comb, make_cross, make_disk, make_ring
from src.metrics import crossing_number, count_holes, skeleton_metrics
from src.skeleton import zhang_suen


def test_crossing_number_endpoint_and_line():
    line = np.zeros((5, 5), np.uint8)
    line[2, 1:4] = 255

    A = crossing_number(line)
    assert A[2, 1] == 1      # 端点
    assert A[2, 2] == 2      # 普通点
    assert A[2, 3] == 1      # 端点


def test_crossing_number_branch():
    cross = np.zeros((7, 7), np.uint8)
    cross[3, 1:6] = 255
    cross[1:6, 3] = 255

    A = crossing_number(cross)
    assert A[3, 3] == 4      # 四路分支


def test_skeleton_metrics_comb():
    binary = make_comb()
    m = skeleton_metrics(binary, zhang_suen(binary))

    assert m["endpoints"] == 7
    assert m["branch_points"] == 5
    assert m["components"] == 1
    assert m["skeleton_holes"] == 0


def test_skeleton_metrics_ring_topology():
    binary = make_ring()
    m = skeleton_metrics(binary, zhang_suen(binary))

    assert m["endpoints"] == 0
    assert m["branch_points"] == 0
    assert m["components"] == 1
    assert m["skeleton_holes"] == 1


def test_skeleton_metrics_cross():
    binary = make_cross()
    m = skeleton_metrics(binary, zhang_suen(binary))

    assert m["endpoints"] == 4
    assert m["branch_points"] == 1


def test_skeleton_metrics_disk_isolated():
    binary = make_disk()
    m = skeleton_metrics(binary, zhang_suen(binary))

    assert m["skeleton_pixels"] == 1
    assert m["isolated_points"] == 1


def test_count_holes():
    assert count_holes(make_ring()) == 1
    assert count_holes(make_disk()) == 0