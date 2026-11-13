"""合成二值测试图形生成模块。

用途：生成几何真值已知的二值图，用来验证二值化、距离变换和骨架提取算法。
约定：前景 = 255，背景 = 0，单通道二维数组。
"""
from pathlib import Path

import cv2
import numpy as np

DEFAULT_SIZE = 256


def _canvas(size):
    """创建全背景（0）画布。"""
    return np.zeros((size, size), np.uint8)


def _center(size, center):
    """未指定中心时使用画布中心。"""
    if center is None:
        return size // 2, size // 2
    return int(center[0]), int(center[1])


def make_disk(size=DEFAULT_SIZE, radius=60, center=None):
    """实心圆盘。中轴理论上是圆心处的一个点。"""
    img = _canvas(size)
    cv2.circle(img, _center(size, center), radius, 255, -1)
    return img


def make_rectangle(size=DEFAULT_SIZE, width=140, height=60, center=None):
    """实心矩形。中轴理论上是沿长轴的中线段。"""
    img = _canvas(size)
    cx, cy = _center(size, center)
    x0, y0 = cx - width // 2, cy - height // 2
    # 注意：cv2.rectangle 的坐标包含端点，所以要 -1 才能得到精确的 width×height
    cv2.rectangle(img, (x0, y0), (x0 + width - 1, y0 + height - 1), 255, -1)
    return img


def make_ellipse(size=DEFAULT_SIZE, axes=(100, 45), angle=0, center=None):
    """实心椭圆。axes = (长半轴, 短半轴)。中轴是沿长轴的中线段。"""
    img = _canvas(size)
    cv2.ellipse(img, _center(size, center), axes, angle, 0, 360, 255, -1)
    return img


def make_cross(size=DEFAULT_SIZE, arm=90, thickness=30, center=None):
    """十字形。中轴理论上有 4 个端点、1 个四度分支点。"""
    img = _canvas(size)
    cx, cy = _center(size, center)
    half = thickness // 2
    cv2.rectangle(img, (cx - arm, cy - half), (cx + arm, cy + half), 255, -1)
    cv2.rectangle(img, (cx - half, cy - arm), (cx + half, cy + arm), 255, -1)
    return img


def make_l_shape(size=DEFAULT_SIZE, long_arm=110, short_arm=50, thickness=24, center=None):
    """L 形。中轴是一条折线：2 个端点、0 个分支点。"""
    img = _canvas(size)
    cx, cy = _center(size, center)
    x0, y0 = cx - long_arm // 2, cy - long_arm // 2
    cv2.rectangle(img, (x0, y0), (x0 + thickness - 1, y0 + long_arm - 1), 255, -1)
    cv2.rectangle(img, (x0, y0 + long_arm - thickness),
                  (x0 + short_arm - 1, y0 + long_arm - 1), 255, -1)
    return img


def make_ring(size=DEFAULT_SIZE, outer_radius=90, inner_radius=60, center=None):
    """圆环。中轴是一个闭合圆：0 个端点、1 个孔洞。用于验证拓扑保持。"""
    img = _canvas(size)
    cv2.circle(img, _center(size, center), outer_radius, 255, -1)
    cv2.circle(img, _center(size, center), inner_radius, 0, -1)
    return img


def make_comb(size=DEFAULT_SIZE, teeth=5, tooth_length=55, tooth_width=12,
              spine_width=18, margin=30, side_pad=20):
    """梳状图。多分支结构，用来观察伪分支和短枝。"""
    img = _canvas(size)
    y_bottom = size - margin
    y_spine = y_bottom - spine_width
    img[y_spine:y_bottom, margin:size - margin] = 255      # 横脊
    usable = (size - 2 * margin) - 2 * side_pad - tooth_width
    step = usable / max(teeth - 1, 1)
    for i in range(teeth):
        x = margin + side_pad + int(round(i * step))
        img[y_spine - tooth_length:y_spine, x:x + tooth_width] = 255   # 齿
    return img


# 名字 → 生成函数，方便批量生成
SHAPE_MAKERS = {
    "disk": make_disk,
    "rectangle": make_rectangle,
    "ellipse": make_ellipse,
    "cross": make_cross,
    "l_shape": make_l_shape,
    "ring": make_ring,
    "comb": make_comb,
}


def generate_all(output_dir, shape_names=None, size=DEFAULT_SIZE):
    """生成并保存所有（或指定）合成图形，返回保存的路径列表。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(SHAPE_MAKERS) if shape_names is None else list(shape_names)
    saved = []
    for name in names:
        if name not in SHAPE_MAKERS:
            raise KeyError(f"未知图形:{name}，可选:{list(SHAPE_MAKERS)}")
        image = SHAPE_MAKERS[name](size=size)
        path = out_dir / f"{name}.png"
        if not cv2.imwrite(str(path), image):
            raise IOError(f"保存失败:{path}")
        saved.append(path)
    return saved