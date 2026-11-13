"""细化算法的单元测试。"""
import numpy as np
import pytest

from src.dataset import make_cross, make_rectangle, make_ring
from src.skeleton import zhang_suen, thin_opencv


@pytest.mark.parametrize("maker, expected", [
    (make_rectangle, 80),
    (make_ring, 476),
    (make_cross, 299),
])
def test_zhang_suen_pixel_counts(maker, expected):
    skeleton = zhang_suen(maker())
    assert int((skeleton == 255).sum()) == expected


def test_zhang_suen_matches_opencv():
    for maker in (make_rectangle, make_ring, make_cross):
        binary = maker()
        ours = zhang_suen(binary)
        cv_zs = thin_opencv(binary, "zhang-suen")
        assert np.array_equal(ours, cv_zs)


def test_thin_opencv_invalid_method():
    with pytest.raises(ValueError):
        thin_opencv(make_rectangle(), "not-a-method")