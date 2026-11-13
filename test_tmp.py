from pathlib import Path

import cv2
import numpy as np

from src.preprocessing import (
    load_image,
    binarize_image,
    compute_distance_transform,
    select_largest_component,
    remove_small_components,
)


def make_disk(shape, center, radius):
    """生成实心圆盘二值图：圆内 255，圆外 0。"""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    inside = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius ** 2
    return inside.astype(np.uint8) * 255


# ==================== 测试 1：合成圆盘 ====================
print("=" * 55)
print("测试 1：合成圆盘（半径 40）")
disk = make_disk((200, 200), (100, 100), 40)
print("disk 自检 -> dtype:", disk.dtype, "| shape:", disk.shape,
      "| 唯一值:", np.unique(disk))
assert disk.dtype == np.uint8
assert set(np.unique(disk).tolist()) == {0, 255}

d = compute_distance_transform(disk, mask_size=cv2.DIST_MASK_PRECISE)
print("最大值:", round(float(d.max()), 3), "（理论 40）")
print("圆心值:", round(float(d[100, 100]), 3), "（理论 40）")
assert abs(d.max() - 40) < 1.0
assert abs(d[100, 100] - 40) < 1.0

# ==================== 测试 2：合成矩形 ====================
print("=" * 55)
print("测试 2：合成矩形（100×40）")
rect = np.zeros((200, 200), np.uint8)
rect[80:120, 50:150] = 255          # 高 40、宽 100
d2 = compute_distance_transform(rect, mask_size=cv2.DIST_MASK_PRECISE)
print("最大值:", round(float(d2.max()), 3), "（理论 20）")
assert abs(d2.max() - 20) < 1.0

# ==================== 测试 3：真实图像 ====================
print("=" * 55)
print("测试 3：真实图像（暗物体方案）")
image = load_image(r"C:\Users\User\Desktop\nailong.jpeg")
binary = binarize_image(image, threshold=127, invert=True)
dist = compute_distance_transform(binary)
print("dtype:", dist.dtype)
print("shape:", dist.shape)
print("背景是否全为 0:", bool(np.all(dist[binary == 0] == 0)))
print("前景是否全 > 0:", bool(np.all(dist[binary == 255] > 0)))
print("最大距离:", round(float(dist.max()), 3))

np.save(r"E:\PythonProject\skeleton-eval\_tmp_dist.npy", dist)
dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
cv2.imwrite(r"E:\PythonProject\skeleton-eval\_tmp_dist_vis.png", dist_vis)
heat = cv2.applyColorMap(dist_vis, cv2.COLORMAP_JET)
cv2.imwrite(r"E:\PythonProject\skeleton-eval\_tmp_dist_heat.png", heat)

# ==================== 测试 4：mask_size 对比 ====================
print("=" * 55)
print("测试 4：mask_size 对比")
for ms in (3, 5, cv2.DIST_MASK_PRECISE):
    dm = compute_distance_transform(binary, mask_size=ms)
    print(f"mask_size={ms:>2}: 最大值 = {float(dm.max()):.3f}")

# ==================== 测试 5：负向测试 ====================
print("=" * 55)
print("测试 5：负向测试")
cases = [
    (None, ValueError),
    ([1, 2, 3], TypeError),
    (np.zeros((10, 10, 3), np.uint8), ValueError),
    (np.zeros((0, 0), np.uint8), ValueError),
    (np.zeros((10, 10), np.float32), TypeError),
    (np.zeros((10, 10), np.uint8), ValueError),          # 无前景
    (np.ones((10, 10), np.uint8) * 128, ValueError),     # 非二值
]
for bad, expect in cases:
    try:
        compute_distance_transform(bad)
        print(f"❌ {type(bad).__name__} 没有抛出异常")
    except expect as e:
        print(f"✅ {type(e).__name__}: {str(e)[:60]}")

# ==================== 测试 6：真实图像物体选择 ====================
print("=" * 55)
print("测试 6：真实图像（亮度带提取指甲主体）")

image = load_image(r"C:\Users\User\Desktop\nailong.jpeg")

# 指甲主体是亮度 236~250 的浅灰实心区域（背景为 255，皮肤约 222）
band = ((image >= 236) & (image <= 250)).astype(np.uint8) * 255
print("带阈值前景像素:", int((band == 255).sum()))

num_labels = cv2.connectedComponentsWithStats(band, connectivity=8)[0]
print("连通域数量:", num_labels - 1)

nail = select_largest_component(band)
print("最大连通域面积:", int((nail == 255).sum()))

dist_nail = compute_distance_transform(nail)
print("指甲距离变换最大值:", round(float(dist_nail.max()), 2), "（预期 119.1）")

# 对比：只做 min_area 过滤是另一种去噪方式
cleaned = remove_small_components(band, min_area=100)
print("min_area=100 过滤后像素:", int((cleaned == 255).sum()))

real_dir = Path(r"E:\PythonProject\skeleton-eval\data\input\real")
real_dir.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(real_dir / "nail_body.png"), nail)
print("已保存:", real_dir / "nail_body.png")