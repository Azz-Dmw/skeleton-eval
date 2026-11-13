"""预处理模块的单元测试。"""
import cv2
import numpy as np
import pytest

from src.dataset import make_disk
from src.preprocessing import (
    load_image,
    binarize_image,
    compute_distance_transform,
    select_largest_component,
    remove_small_components,
    fill_small_holes,
)


def test_load_image_returns_gray_ndarray(tmp_path):
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), make_disk(size=64, radius=20))

    image = load_image(path)

    assert isinstance(image, np.ndarray)
    assert image.dtype == np.uint8
    assert image.ndim == 2
    assert image.shape == (64, 64)


def test_load_image_missing_file():
    with pytest.raises(FileNotFoundError):
        load_image("definitely_not_here.png")


def test_load_image_wrong_type():
    with pytest.raises(TypeError):
        load_image(123)


def test_binarize_image_output_values():
    gray = np.array([[0, 100, 200, 255]], np.uint8)
    binary = binarize_image(gray, threshold=127)

    assert binary.dtype == np.uint8
    assert set(np.unique(binary).tolist()) == {0, 255}
    assert binary[0, 0] == 0
    assert binary[0, 1] == 0
    assert binary[0, 2] == 255
    assert binary[0, 3] == 255


def test_binarize_invert_swaps():
    gray = np.array([[0, 255]], np.uint8)
    normal = binarize_image(gray, 127, invert=False)
    inverted = binarize_image(gray, 127, invert=True)
    assert np.array_equal(inverted, 255 - normal)


def test_binarize_invalid():
    with pytest.raises(ValueError):
        binarize_image(None)
    with pytest.raises(TypeError):
        binarize_image([1, 2, 3])
    with pytest.raises(ValueError):
        binarize_image(np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ValueError):
        binarize_image(np.zeros((0, 0), np.uint8))


def test_distance_transform_disk_radius():
    disk = make_disk(size=200, radius=40)
    dist = compute_distance_transform(disk, mask_size=cv2.DIST_MASK_PRECISE)

    assert dist.dtype == np.float32
    assert abs(float(dist.max()) - 40.0) < 1.0
    assert np.all(dist[disk == 0] == 0)
    assert np.all(dist[disk == 255] > 0)


def test_distance_transform_rejects_non_binary():
    with pytest.raises(ValueError):
        compute_distance_transform(np.full((8, 8), 128, np.uint8))


def test_select_largest_component_keeps_biggest():
    binary = np.zeros((10, 10), np.uint8)
    binary[0:2, 0:2] = 255       # 4 像素
    binary[5:9, 5:9] = 255       # 16 像素

    result = select_largest_component(binary)
    assert int((result == 255).sum()) == 16


def test_remove_small_components():
    binary = np.zeros((10, 10), np.uint8)
    binary[0, 0] = 255
    binary[5:9, 5:9] = 255

    result = remove_small_components(binary, min_area=4)
    assert int((result == 255).sum()) == 16


def test_fill_small_holes():
    binary = np.zeros((20, 20), np.uint8)
    binary[2:18, 2:18] = 255
    binary[8, 8] = 0             # 1 像素孔洞

    filled = fill_small_holes(binary, max_area=4)
    assert filled[8, 8] == 255