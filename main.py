import cv2
from src.preprocessing import load_image

#测试
if __name__ == "__main__":
    #读取图像
    image_path = r"C:\Users\User\Desktop\nailong.jpeg"
    image = load_image(image_path)

    print("类型:",type(image))
    print("dtype:",image.dtype)
    print("shape:",image.shape)

    #反向测试 1:轮径不存在
    try:
        load_image(r"C:\Users\User\Desktop\not_exist_123.jpeg")
    except FileNotFoundError as e:
        print("测试1通过:",e)

    # 反向测试 2：传入非路径对象
    try:
        load_image(image)
    except TypeError as e:
        print("测试2通过:",e)

    raw = image     #原灰度图
    _,bw = cv2.threshold(raw,127,255,cv2.THRESH_BINARY)
    fg = int((bw == 255).sum())
    total = int(bw.size)
    print(f"阈值为255的像素: {fg}")
    print(f"总像素: {total}")
    print(f"'前景'占比: {fg / total:.2%}")
    cv2.imwrite(r"E:\PythonProject\skeleton-eval\_tmp_binarize_probe.png", bw)
