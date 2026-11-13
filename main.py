"""项目统一入口：单张或批量图像的骨架提取与量化对比。

用法示例：
    python main.py --input data/input/synthetic --binary
    python main.py --input data/input/real/nail_body.png --binary
    python main.py --input data/input/synthetic --method dt
    python main.py --input "C:\\Users\\User\\Desktop\\nailong.jpeg"

注意：--binary 表示输入已经是 {0,255} 的二值图（跳过阈值二值化）。
合成测试图和 nail_body.png 都是二值图，必须加 --binary。
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from src.distance_skeleton import distance_skeleton
from src.metrics import skeleton_metrics
from src.preprocessing import (
    binarize_image,
    fill_small_holes,
    load_image,
    validate_binary_image,
)
from src.skeleton import thin_opencv, zhang_suen

PROJECT_ROOT = Path(__file__).resolve().parent
METHOD_CHOICES = ("all", "zs", "gh", "dt")
IMAGE_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_images(input_path):
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = []
        for pattern in IMAGE_PATTERNS:
            files.extend(sorted(path.glob(pattern)))
        if not files:
            raise FileNotFoundError(f"目录中没有图像文件:{path}")
        return files
    raise FileNotFoundError(f"输入路径不存在:{path}")


def prepare_binary(image, is_binary, cfg):
    """把输入统一处理成清理过孔洞的二值图。"""
    if is_binary:
        validate_binary_image(image)
        binary = image
    else:
        values = np.unique(image)
        if np.all(np.isin(values, (0, 255))):
            print("  [提示] 输入已是二值图但未加 --binary，将按 config 的阈值/极性处理，可能得到相反结果。")

        pre = cfg["preprocessing"]
        binary = binarize_image(image, threshold=pre["threshold"], invert=pre["invert"])

    return fill_small_holes(binary, max_area=cfg["preprocessing"]["max_hole_area"])


def build_skeletons(binary, method, cfg):
    skeletons = {}
    if method in ("all", "zs"):
        skeletons["zhang_suen"] = zhang_suen(binary, max_iter=cfg["thinning"]["max_iter"])
    if method in ("all", "gh"):
        skeletons["guo_hall"] = thin_opencv(binary, "guo-hall")
    if method in ("all", "dt"):
        skeletons["distance_transform"] = distance_skeleton(
            binary, **cfg["distance_skeleton"]
        )
    return skeletons


def main():
    parser = argparse.ArgumentParser(description="二值图像骨架提取与算法评估")
    parser.add_argument("--input", required=True, help="图像文件或目录")
    parser.add_argument("--method", choices=METHOD_CHOICES, default="all",
                        help="zs=Zhang-Suen, gh=Guo-Hall, dt=距离变换, all=全部")
    parser.add_argument("--binary", action="store_true",
                        help="输入已经是二值图（跳过阈值二值化）")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.json"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs"))
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_dir = Path(args.output)
    skel_dir = output_dir / "skeletons"
    report_dir = output_dir / "reports"
    skel_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(args.input)
    rows = []

    for path in images:
        gray = load_image(path)
        binary = prepare_binary(gray, args.binary, cfg)
        skeletons = build_skeletons(binary, args.method, cfg)

        print(f"\n[{path.name}]  前景像素={int((binary == 255).sum())}")
        print(f"  {'method':<20}{'px':>6}{'endp':>6}{'branch':>7}{'comp':>6}{'holes':>6}")

        for name, skel in skeletons.items():
            metrics = skeleton_metrics(binary, skel)
            print(f"  {name:<20}{metrics['skeleton_pixels']:>6}{metrics['endpoints']:>6}"
                  f"{metrics['branch_points']:>7}{metrics['components']:>6}"
                  f"{metrics['skeleton_holes']:>6}")

            cv2.imwrite(str(skel_dir / f"{path.stem}_{name}.png"), skel)
            rows.append({"image": path.stem, "method": name, **metrics})

    fields = ["image", "method", "skeleton_pixels", "endpoints", "branch_points",
              "branch_pixels", "components", "isolated_points", "skeleton_holes",
              "object_holes", "max_radius", "mean_radius"]
    csv_path = report_dir / "metrics_main.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n共处理 {len(images)} 张图，{len(rows)} 条记录")
    print("骨架保存目录:", skel_dir)
    print("指标 CSV:", csv_path)


if __name__ == "__main__":
    main()