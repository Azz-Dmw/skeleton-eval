"""批量统计骨架指标并保存 CSV。

运行：python run_metrics.py
输出：outputs/reports/metrics.csv
"""
import csv
from pathlib import Path

import cv2

from src.metrics import skeleton_metrics
from src.skeleton import zhang_suen, thin_opencv

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

FIELDS = [
    "image", "method",
    "skeleton_pixels", "endpoints", "branch_points", "branch_pixels",
    "components", "isolated_points", "skeleton_holes", "object_holes",
    "max_radius", "mean_radius",
]


def main():
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

        skeletons = {
            "zhang_suen_ours": zhang_suen(binary),
            "zhang_suen_cv": thin_opencv(binary, "zhang-suen"),
            "guo_hall_cv": thin_opencv(binary, "guo-hall"),
        }

        print(f"\n[{path.stem}]  object_holes={skeleton_metrics(binary, skeletons['zhang_suen_cv'])['object_holes']}")
        print(f"  {'method':<18}{'px':>5}{'endp':>6}{'branch':>7}{'comp':>6}{'holes':>6}")
        for method, skel in skeletons.items():
            m = skeleton_metrics(binary, skel)
            print(f"  {method:<18}{m['skeleton_pixels']:>5}{m['endpoints']:>6}"
                  f"{m['branch_points']:>7}{m['components']:>6}{m['skeleton_holes']:>6}")
            row = {"image": path.stem, "method": method}
            row.update(m)
            rows.append(row)

    csv_path = REPORT_DIR / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n共 {len(rows)} 行，已保存: {csv_path}")


if __name__ == "__main__":
    main()