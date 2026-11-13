from pathlib import Path
import cv2
import numpy as np


#函数 A:读取图像
def load_image(image_path:str|Path):
    """
    以灰度模式读取图像
    :param image_path:图像路径，支持str或者Path
    :return:灰度图像数组(H,W),dtype=uint8
    :raise TypeError():抛出类型异常
    :raise FileNotFoundError():抛出文件异常
    :raise ValueError():抛出类型异常
    """
    #1.参数类型校验+将传入的路径统一转换为Path对象
    if not isinstance(image_path,(str,Path)):
        raise TypeError(f"错误:image_path必须是str或者Path，接收到{type(image_path).__name__}")

    path = Path(image_path)

    #2.检查路径是否存在且为文件(后续统一使用path)
    if not path.exists():
        raise FileNotFoundError(f"错误:图像文件不存在:{path}")

    if not path.is_file():
        raise FileNotFoundError(f"错误:路径不是文件:{path}")

    #3.使用OpenCV以灰度模式读取图像
    image = cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)

    #4.检查图像是否读取成功
    if image is None:
        raise ValueError(f"错误:文件存在，但OpenCV无法读取或解码图像:{path}")

    #5.返回图像数组
    return image

def binarize_image(image:np.ndarray,threshold: int = 127,invert:bool = False) -> np.ndarray:
    """
    将二维灰度图二值化，输出满足"前景=255，背景=0"约定的掩膜。

    :param image: 二维灰度图像数组，dtype 一般为 uint8
    :param threshold: 阈值。翻转前，像素值 > threshold 视为前景
    :param invert: 是否翻转前景/背景极性。白底深色物体（如本项目的指甲图）应传 True
    :return: 二维 uint8 数组，取值仅可能为 {0, 255}
    :raises ValueError: image 为 None，或不是二维数组，或为空数组
    :raises TypeError: image 不是 numpy.ndarray
    """
    # 1. 输入校验：None 单独判断（值为空，更接近 ValueError）
    if image is None:
        raise ValueError("错误:image 不能为None")

    # 2. 输入校验：必须是 numpy 数组
    if not isinstance(image,np.ndarray):
        raise TypeError(f"错误:image 必须是 numpy.ndarray，接收到 {type(image).__name__}")

    # 3. 输入校验：必须是二维数组（灰度图）。彩色图应由调用者先转换
    if image.ndim != 2:
        raise ValueError(f"错误:image 必须是二维灰度图，当前维度为 {image.ndim}")

    # 4. 输入校验：不能是空数组
    if image.size == 0:
        raise ValueError("错误:image 不能是空数组")

    # 5. 布尔比较：得到 (H, W) 的 bool 掩膜。
    #    用比较而不是乘法，避免 uint8 溢出
    mask = image > threshold

    # 6. 极性翻转：只翻转掩膜，不改变比较规则
    if invert:
        mask = ~mask

    # 7. 把 True/False 映射为 255/0，并统一为 uint8
    binary = np.where(mask,255,0).astype(np.uint8)

    return binary

def compute_distance_transform(
        binary_image:np.ndarray,
        distance_type:int = cv2.DIST_L2,
        mask_size:int = 5,
) -> np.ndarray:
    """
        计算二值图像的距离变换。

        对输出数组中的每个像素 p：
          - 若 p 是背景（值为 0），则 DT(p) = 0；
          - 若 p 是前景（值为 255），则 DT(p) = p 到最近背景像素的欧氏距离。
        该距离在几何上等于"以 p 为圆心、完全落在前景内的最大内切圆半径"。

        :param binary_image: 二维二值图像，dtype 必须为 uint8，取值只能为 {0, 255}
        :param distance_type: 距离度量。cv2.DIST_L2（欧氏，默认）、
                              cv2.DIST_L1（曼哈顿）、cv2.DIST_C（棋盘）
        :param mask_size: 距离近似的掩膜尺寸。3（快、误差大）、5（默认）、
                          cv2.DIST_MASK_PRECISE（精确算法，值为 0）
        :return: 与输入同形状的 float32 距离数组
        :raises TypeError: 输入不是 numpy.ndarray，或 dtype 不是 uint8
        :raises ValueError: 输入为 None、非二维、空数组、非二值、无前景像素，
                            或 distance_type / mask_size 取值不支持
        """
    # ---------- 第一组：输入数据的基本校验 ----------
    if binary_image is None:
        raise ValueError("错误:binary_image 不能为 None")

    if not isinstance(binary_image,np.ndarray):
        raise TypeError(f"错误:binary_image 必须是 numpy.ndarray，接收到 {type(binary_image).__name__}")

    if binary_image.ndim != 2:
        raise ValueError( f"错误:binary_image 必须是二维数组，当前维度为 {binary_image.ndim}")

    if binary_image.size == 0:
        raise ValueError("错误:binary_image 不能是空数组")

    if binary_image.dtype != np.uint8:
        raise TypeError(f"错误:binary_image 的 dtype 必须是 uint8，当前为 {binary_image.dtype}；"
            "如果传入的是布尔掩膜，请先用 .astype(np.uint8) * 255 转换")

    # ---------- 第二组：二值语义校验 ----------
    unique_values = np.unique(binary_image)
    if not np.all(np.isin(unique_values,(0,255))):
        raise ValueError(
            "错误:binary_image 必须是二值图像（取值只能是 0 和 255），"
            f"当前出现的取值为 {unique_values.tolist()}"
        )

    if not np.any(binary_image == 255):
        raise ValueError("错误:binary_image 中没有前景像素（全为 0），无法计算距离变换")

    # ---------- 第三组：算法参数校验 ----------
    if distance_type not in (cv2.DIST_L1,cv2.DIST_L2,cv2.DIST_C):
        raise ValueError(
            f"错误:不支持的 distance_type={distance_type}，"
            "可选 cv2.DIST_L1 / cv2.DIST_L2 / cv2.DIST_C"
        )

    if mask_size not in (3,5,cv2.DIST_MASK_PRECISE):
        raise ValueError(
            f"错误:不支持的 mask_size={mask_size}，"
            "可选 3 / 5 / cv2.DIST_MASK_PRECISE"
        )

    # ---------- 第四组：调用 OpenCV 完成计算 ----------
    #前面代码都是校验，这里才是具体功能实现
    dist = cv2.distanceTransform(binary_image,distance_type,mask_size)

    return dist

def _validate_binary_image(binary_image:np.ndarray) -> None:
    """私有辅助函数：校验二值图输入，供本模块多个函数复用。"""
    if binary_image is None:
        raise ValueError("错误:binary_image 不能为 None")

    if not isinstance(binary_image, np.ndarray):
        raise TypeError(
            f"错误:binary_image 必须是 numpy.ndarray，接收到 {type(binary_image).__name__}"
        )

    if binary_image.ndim != 2:
        raise ValueError(
            f"错误:binary_image 必须是二维数组，当前维度为 {binary_image.ndim}"
        )

    if binary_image.size == 0:
        raise ValueError("错误:binary_image 不能是空数组")

    if binary_image.dtype != np.uint8:
        raise TypeError(
            f"错误:binary_image 的 dtype 必须是 uint8，当前为 {binary_image.dtype}"
        )

    unique_values = np.unique(binary_image)
    if not np.all(np.isin(unique_values, (0, 255))):
        raise ValueError(
            "错误:binary_image 必须是二值图像（取值只能是 0 和 255），"
            f"当前出现的取值为 {unique_values.tolist()}"
        )

def select_largest_component(binary_image: np.ndarray) -> np.ndarray:
    """
    保留面积最大的前景连通域，其余前景置 0。

    使用 8 连通（对角线接触也算相连），避免细线被切断。
    输出仍为 {0, 255} 的 uint8 二值图。
    """
    _validate_binary_image(binary_image)

    num_labels,labels,stats,_= cv2.connectedComponentsWithStats(binary_image,connectivity=8)

    if num_labels <= 1:
        raise ValueError("错误:binary_image 中没有前景连通域")

    areas = stats[1:,cv2.CC_STAT_AREA]      # 跳过标签 0（背景）
    largest_label = 1 + int(np.argmax(areas))

    return (labels == largest_label).astype(np.uint8) * 255

def remove_small_components(binary_image: np.ndarray, min_area: int = 100) -> np.ndarray:
    """
    删除面积小于 min_area 的前景连通域（去噪点）。

    与 select_largest_component 的区别：这里保留所有"够大"的连通域，
    适合一张图里有多个研究对象的场景。
    """
    _validate_binary_image(binary_image)
    if min_area < 0:
        raise ValueError(f"错误:min_area 不能为负数，当前为 {min_area}")

    num_labels,labels,stats,_ = cv2.connectedComponentsWithStats(binary_image,connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(binary_image)

    keep = stats[1:,cv2.CC_STAT_AREA] >= min_area
    keep_labels = np.nonzero(keep)[0] + 1        # 数组下标 → 真实标签

    return np.isin(labels,keep_labels).astype(np.uint8) * 255


