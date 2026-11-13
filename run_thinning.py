"""批量运行细化算法，保存骨架和对比图。

运行：python run_thinning.py
输出：outputs/skeletons/{图像名}_{算法}.png
      outputs/skeletons/{图像名}_compare.png  （原图 + 三种骨架并排）
"""
import time
from pathlib import Path

import cv2
import numpy as np

from src.skeleton import zhang_suen, thin_opencv

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "skeletons"


def _label(gray_img, text):
    """给单通道图顶部加黑边并写英文标签（Hershey 字体不支持中文）。"""
    bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    bgr = cv2.copyMakeBorder(bgr, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(bgr, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def process(name, image):
    """对一张图运行三种细化，保存结果并打印统计。"""
    t0 = time.perf_counter()
    ours = zhang_suen(image)
    t1 = time.perf_counter()
    cv_zs = thin_opencv(image, "zhang-suen")
    t2 = time.perf_counter()
    cv_gh = thin_opencv(image, "guo-hall")
    t3 = time.perf_counter()

    n_ours = int((ours == 255).sum())
    n_zs = int((cv_zs == 255).sum())
    n_gh = int((cv_gh == 255).sum())
    diff = int(np.count_nonzero(ours != cv_zs))

    print(f"{name:<12} ours={n_ours:>4}px/{1000 * (t1 - t0):>7.1f}ms  "
          f"cvZS={n_zs:>4}px/{1000 * (t2 - t1):>6.1f}ms  "
          f"cvGH={n_gh:>4}px/{1000 * (t3 - t2):>6.1f}ms  diff={diff}")

    # 保存三种骨架（阶段三统计指标时直接读这些文件）
    for suffix, skel in (("ours", ours), ("cv_zhangsuen", cv_zs), ("cv_guohall", cv_gh)):
        cv2.imwrite(str(OUTPUT_DIR / f"{name}_{suffix}.png"), skel)

    # 并排对比图：原图 | 我们的 ZS | OpenCV ZS | OpenCV GH
    montage = np.hstack([
        _label(image, "original"),
        _label(ours, "ours ZS"),
        _label(cv_zs, "cv ZS"),
        _label(cv_gh, "cv GH"),
    ])
    cv2.imwrite(str(OUTPUT_DIR / f"{name}_compare.png"), montage)
    return diff


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted((INPUT_DIR / "synthetic").glob("*.png"))
    real_path = INPUT_DIR / "real" / "nail_body.png"
    if real_path.is_file():
        paths.append(real_path)

    total_diff = 0
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"读取失败:{path}")
        total_diff += process(path.stem, image)

    print("-" * 60)
    print(f"共处理 {len(paths)} 张图；ours 与 OpenCV ZS 的总差异像素 = {total_diff}")