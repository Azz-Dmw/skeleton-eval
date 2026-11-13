"""骨架提取算法（细化类）。

当前包含：
- zhang_suen：自己实现的 Zhang-Suen 细化
- thin_opencv：OpenCV ximgproc 基准实现（Zhang-Suen / Guo-Hall）
"""
import cv2
import numpy as np

from src.preprocessing import validate_binary_image


def zhang_suen(binary_image: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """
    Zhang-Suen 细化算法（两次子迭代轮流删除边界像素）。

    :param binary_image: 二维二值图，取值 {0, 255}
    :param max_iter: 最大迭代次数，防止病态输入死循环
    :return: 细化后的二值图，取值 {0, 255}，dtype uint8
    """
    validate_binary_image(binary_image)
    if max_iter < 1:
        raise ValueError(f"错误:max_iter 必须 >= 1，当前为 {max_iter}")

    img = (binary_image > 0).astype(np.uint8)

    for _ in range(max_iter):
        changed = False

        for step in (0, 1):
            # 每轮子迭代都基于"当前"的 img 重新取邻域
            p = np.pad(img, 1, mode="constant", constant_values=0)
            P2 = p[0:-2, 1:-1]      # 上
            P3 = p[0:-2, 2:]        # 右上
            P4 = p[1:-1, 2:]        # 右
            P5 = p[2:, 2:]          # 右下
            P6 = p[2:, 1:-1]        # 下
            P7 = p[2:, 0:-2]        # 左下
            P8 = p[1:-1, 0:-2]      # 左
            P9 = p[0:-2, 0:-2]      # 左上

            # 判据 1：邻域前景像素个数
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9

            # 判据 2：0→1 跳变次数（必须 astype(int)，布尔相加会变成逻辑或）
            A = (((P2 == 0) & (P3 == 1)).astype(np.uint8)
                 + ((P3 == 0) & (P4 == 1)).astype(np.uint8)
                 + ((P4 == 0) & (P5 == 1)).astype(np.uint8)
                 + ((P5 == 0) & (P6 == 1)).astype(np.uint8)
                 + ((P6 == 0) & (P7 == 1)).astype(np.uint8)
                 + ((P7 == 0) & (P8 == 1)).astype(np.uint8)
                 + ((P8 == 0) & (P9 == 1)).astype(np.uint8)
                 + ((P9 == 0) & (P2 == 1)).astype(np.uint8))

            # 判据 3、4：两个子迭代的附加条件
            if step == 0:
                cond = ((B >= 2) & (B <= 6) & (A == 1)
                        & ((P2 * P4 * P6) == 0) & ((P4 * P6 * P8) == 0))
            else:
                cond = ((B >= 2) & (B <= 6) & (A == 1)
                        & ((P2 * P4 * P8) == 0) & ((P2 * P6 * P8) == 0))

            cond &= (img == 1)          # 只处理当前还是前景的像素
            if not cond.any():
                continue
            img[cond] = 0               # 先标记、再统一删除
            changed = True

        if not changed:                 # 两轮子迭代都没有删除 → 收敛
            break

    return img * 255


def thin_opencv(binary_image: np.ndarray, method: str = "zhang-suen") -> np.ndarray:
    """
    OpenCV 基准细化实现（cv2.ximgproc.thinning）。

    :param binary_image: 二维二值图，取值 {0, 255}
    :param method: "zhang-suen" 或 "guo-hall"
    :return: 细化后的二值图，取值 {0, 255}
    """
    validate_binary_image(binary_image)

    if method == "zhang-suen":
        thinning_type = cv2.ximgproc.THINNING_ZHANGSUEN
    elif method == "guo-hall":
        thinning_type = cv2.ximgproc.THINNING_GUOHALL
    else:
        raise ValueError(f'错误:不支持的 method={method!r}，可选 "zhang-suen" / "guo-hall"')

    return cv2.ximgproc.thinning(binary_image, thinningType=thinning_type)