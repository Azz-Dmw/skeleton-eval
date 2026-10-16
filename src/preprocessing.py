from pathlib import Path
import cv2

#函数 A:读取图像
def load_imgae(image_path:str|Path)->"cv2.typing.MatLike":
    """
    以灰度模式读取图像
    :param image_path:图像路径，支持str或者Path
    :return:灰度图像数组(H,W),dtype=uint8
    :raise TypeError():抛出类型异常
    :raise FileNotFoundError():抛出文件异常
    :raise ValueError():抛出数据异常
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