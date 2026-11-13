"""距离变换骨架实验：与 Zhang-Suen / Guo-Hall 对比。

运行：python run_distance_skeleton.py
输出：
  outputs/skeletons/{图像名}_dt.png
  outputs/skeletons/{图像名}_compare_dt.png
  outputs/reports/metrics_comparison.csv
"""
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from src.distance_skeleton import distance_skeleton
from src.metrics import skeleton_metrics
from src.preprocessing import fill_small_holes
from src.skeleton import zhang_suen, thin_opencv

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
SKEL_DIR = PROJECT_ROOT / "outputs" / "skeletons"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

FIELDS = ["image", "method", "time_ms", "skeleton_pixels", "endpoints",
          "branch_points", "branch_pixels", "components", "isolated_points",
          "skeleton_holes", "object_holes", "max_radius", "mean_radius"]


def _label(img, text):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.copyMakeBorder(img, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    SKEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted((INPUT_DIR / "synthetic").glob("*.png"))
    real = INPUT_DIR / "real" / "nail_body.png"
    if real.is_file():
        paths.append(real)

    rows = []
    for path in paths:
        binary = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if binary is None:
            raise ValueError(f"读取失败:{path}")
        binary = fill_small_holes(binary)

        t0 = time.perf_counter()
        dt_skel = distance_skeleton(binary)
        t1 = time.perf_counter()

        results = {
            "distance_transform": (dt_skel, (t1 - t0) * 1000),
            "zhang_suen": (zhang_suen(binary), None),
            "guo_hall": (thin_opencv(binary, "guo-hall"), None),
        }

        print(f"\n[{path.stem}]")
        print(f"  {'method':<20}{'px':>5}{'endp':>6}{'branch':>7}{'comp':>6}{'holes':>6}")
        for method, (skel, t_ms) in results.items():
            m = skeleton_metrics(binary, skel)
            tail = f"   {t_ms:.1f}ms" if t_ms is not None else ""
            print(f"  {method:<20}{m['skeleton_pixels']:>5}{m['endpoints']:>6}"
                  f"{m['branch_points']:>7}{m['components']:>6}{m['skeleton_holes']:>6}{tail}")

            row = {"image": path.stem, "method": method,
                   "time_ms": round(t_ms, 2) if t_ms is not None else ""}
            row.update(m)
            rows.append(row)

        # 并排对比图：二值图 | 距离热力图 | DT 骨架 | ZS | GH
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        heat = cv2.applyColorMap(
            cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        montage = np.hstack([
            _label(binary, "binary"),
            _label(heat, "distance"),
            _label(dt_skel, "DT skeleton"),
            _label(results["zhang_suen"][0], "ZS"),
            _label(results["guo_hall"][0], "GH"),
        ])
        cv2.imwrite(str(SKEL_DIR / f"{path.stem}_dt.png"), dt_skel)
        cv2.imwrite(str(SKEL_DIR / f"{path.stem}_compare_dt.png"), montage)

    csv_path = REPORT_DIR / "metrics_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n共 {len(rows)} 行，已保存: {csv_path}")


if __name__ == "__main__":
    main()