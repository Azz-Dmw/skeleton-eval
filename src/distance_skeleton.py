"""基于距离变换的骨架提取（中轴近似）。

流程：
1. 计算距离变换；
2. 在平滑后的距离图上做方向性非极大值抑制，得到脊线；
3. 把在分支点处断开的脊线片段用最近邻直线连接；
4. 细化成单像素宽，并限制在物体内部。

注意：这是对严格中轴（medial axis）的离散近似，不是数学意义上的精确中轴。
"""
import cv2
import numpy as np

from src.preprocessing import validate_binary_image


def ridge_mask(binary_image: np.ndarray, sigma: float = 1.5,
               tolerance: float = 0.3, min_distance: float = 1.0):
    """
    方向性非极大值抑制，提取距离变换的脊线。

    对每个像素，沿距离函数梯度方向前后各取一个采样点；
    若该像素的值不小于这两个采样点（允许 tolerance 的容差），则认为它在脊线上。

    :return: (脊线掩膜, 距离变换数组)
    """
    validate_binary_image(binary_image)
    if sigma <= 0:
        raise ValueError(f"错误:sigma 必须 > 0，当前为 {sigma}")
    if tolerance < 0:
        raise ValueError(f"错误:tolerance 不能为负数，当前为 {tolerance}")
    if min_distance < 0:
        raise ValueError(f"错误:min_distance 不能为负数，当前为 {min_distance}")

    dist = cv2.distanceTransform(binary_image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    smooth = cv2.GaussianBlur(dist, (0, 0), sigma)

    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    norm = np.sqrt(gx * gx + gy * gy) + 1e-9
    ux, uy = gx / norm, gy / norm

    h, w = smooth.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    forward = cv2.remap(smooth, grid_x + ux, grid_y + uy,
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    backward = cv2.remap(smooth, grid_x - ux, grid_y - uy,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    ridge = ((smooth >= forward - tolerance) & (smooth >= backward - tolerance)
             & (dist > 0) & (dist >= min_distance))

    return ridge.astype(np.uint8) * 255, dist


def bridge_components(mask: np.ndarray, max_gap: float = 40.0) -> np.ndarray:
    """
    把脊线片段用最近邻直线连接。

    每个片段寻找距离最近的另一个片段，若间距 <= max_gap 则画一条直线。
    复杂度 O(k^2)，k 为片段数；适合片段数不多的图像。
    """
    n, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    out = mask.copy()
    if n <= 2:
        return out

    comps = [np.argwhere(labels == i) for i in range(1, n)]
    for i in range(len(comps)):
        best, best_d = None, np.inf
        for j in range(len(comps)):
            if i == j:
                continue
            a, b = comps[i], comps[j]
            d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
            idx = np.unravel_index(np.argmin(d2), d2.shape)
            d = float(np.sqrt(d2[idx]))
            if d < best_d:
                best_d, best = d, (a[idx[0]], b[idx[1]])
        if best is not None and best_d <= max_gap:
            cv2.line(out, (int(best[0][1]), int(best[0][0])),
                     (int(best[1][1]), int(best[1][0])), 255, 1)
    return out


def distance_skeleton(binary_image: np.ndarray, sigma: float = 1.5,
                      tolerance: float = 0.3, min_distance: float = 1.0,
                      max_gap: float = 40.0) -> np.ndarray:
    """
    基于距离变换的骨架提取。

    :param sigma: 距离图高斯平滑的 sigma
    :param tolerance: 脊线判定的容差（必须远小于 1，否则会把整块物体判成脊线）
    :param min_distance: 只保留距离 >= min_distance 的脊线点（抑制边界噪声）
    :param max_gap: 片段连接的最大间距
    :return: 单像素宽的骨架，取值 {0, 255}
    """
    validate_binary_image(binary_image)

    ridge, dist = ridge_mask(binary_image, sigma, tolerance, min_distance)
    linked = bridge_components(ridge, max_gap)
    skel = cv2.ximgproc.thinning(linked, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    skel[dist <= 0] = 0
    return skel