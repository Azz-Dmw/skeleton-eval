"""骨架评估指标。

核心约定
--------
1. 骨架是二维二值图（0/255）。
2. 端点/分支点用"交叉数 A(p)"判定，而不是 8 邻域像素个数。
   A(p) = 8 邻域序列 P2,P3,...,P9,P2 中 0→1 的跳变次数。
   - A = 0：孤立点
   - A = 1：端点
   - A = 2：普通骨架点
   - A >= 3：分支点
   用交叉数可以避免"2-2 阶梯"把普通线段像素误判为度数 4 的分支点。
3. 相邻的分支像素聚成一簇，簇数才是分支点个数。
"""
import cv2
import numpy as np

from src.preprocessing import validate_binary_image


def crossing_number(skeleton: np.ndarray) -> np.ndarray:
    """计算每个像素的交叉数（非骨架像素为 0）。"""
    validate_binary_image(skeleton)

    m = (skeleton > 0).astype(np.uint8)
    p = np.pad(m, 1, mode="constant", constant_values=0)

    P2 = p[0:-2, 1:-1]      # 上
    P3 = p[0:-2, 2:]        # 右上
    P4 = p[1:-1, 2:]        # 右
    P5 = p[2:, 2:]          # 右下
    P6 = p[2:, 1:-1]        # 下
    P7 = p[2:, 0:-2]        # 左下
    P8 = p[1:-1, 0:-2]      # 左
    P9 = p[0:-2, 0:-2]      # 左上

    A = (((P2 == 0) & (P3 == 1)).astype(np.uint8)
         + ((P3 == 0) & (P4 == 1)).astype(np.uint8)
         + ((P4 == 0) & (P5 == 1)).astype(np.uint8)
         + ((P5 == 0) & (P6 == 1)).astype(np.uint8)
         + ((P6 == 0) & (P7 == 1)).astype(np.uint8)
         + ((P7 == 0) & (P8 == 1)).astype(np.uint8)
         + ((P8 == 0) & (P9 == 1)).astype(np.uint8)
         + ((P9 == 0) & (P2 == 1)).astype(np.uint8))

    return A * m


def neighbor_count(skeleton: np.ndarray) -> np.ndarray:
    """8 邻域前景像素个数（仅用于对比，不应用于分支点判定）。"""
    validate_binary_image(skeleton)

    m = (skeleton > 0).astype(np.uint8)
    p = np.pad(m, 1, mode="constant", constant_values=0)
    total = np.zeros_like(m, dtype=np.uint8)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            total += p[1 + dr:p.shape[0] - 1 + dr, 1 + dc:p.shape[1] - 1 + dc]
    return total * m


def endpoint_mask(skeleton: np.ndarray) -> np.ndarray:
    """端点掩膜（A == 1），取值 {0, 255}。"""
    A = crossing_number(skeleton)
    m = (skeleton > 0).astype(np.uint8)
    return ((A == 1) & (m == 1)).astype(np.uint8) * 255


def branch_mask(skeleton: np.ndarray) -> np.ndarray:
    """分支点掩膜（A >= 3），取值 {0, 255}。"""
    A = crossing_number(skeleton)
    m = (skeleton > 0).astype(np.uint8)
    return ((A >= 3) & (m == 1)).astype(np.uint8) * 255


def count_components(binary: np.ndarray) -> int:
    """前景连通分量个数（8 连通）。"""
    validate_binary_image(binary)
    num, _ = cv2.connectedComponents(
        (binary > 0).astype(np.uint8), connectivity=8
    )
    return num - 1


def count_holes(binary: np.ndarray) -> int:
    """
    孔洞个数：不接触图像边界的背景连通分量（背景用 4 连通，防止斜向漏连）。
    """
    validate_binary_image(binary)

    bg = (binary == 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(bg, connectivity=4)

    border = set(labels[0, :].tolist()) | set(labels[-1, :].tolist()) \
        | set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    border.discard(0)

    return num - 1 - len(border)


def skeleton_metrics(binary: np.ndarray, skeleton: np.ndarray) -> dict:
    """
    汇总一张图的骨架指标。

    :param binary: 原二值图（物体，0/255）
    :param skeleton: 该图的骨架（0/255）
    :return: 指标字典
    """
    validate_binary_image(binary)
    validate_binary_image(skeleton)
    if binary.shape != skeleton.shape:
        raise ValueError(
            f"错误:binary 与 skeleton 形状不一致: {binary.shape} vs {skeleton.shape}"
        )

    m = (skeleton > 0).astype(np.uint8)
    A = crossing_number(skeleton)

    endpoints = int(((A == 1) & (m == 1)).sum())
    isolated = int(((A == 0) & (m == 1)).sum())

    branch = ((A >= 3) & (m == 1)).astype(np.uint8)
    num_branch, _ = cv2.connectedComponents(branch, connectivity=8)

    num_comp, _ = cv2.connectedComponents(m, connectivity=8)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    radii = dist[m == 1]

    return {
        "skeleton_pixels": int(m.sum()),
        "endpoints": endpoints,
        "branch_points": num_branch - 1,
        "branch_pixels": int(branch.sum()),
        "components": num_comp - 1,
        "isolated_points": isolated,
        "skeleton_holes": count_holes(skeleton),
        "object_holes": count_holes(binary),
        "max_radius": round(float(dist.max()), 3),
        "mean_radius": round(float(radii.mean()), 3) if radii.size else 0.0,
    }