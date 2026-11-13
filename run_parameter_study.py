"""参数敏感性实验：扫描 sigma 与 tolerance。

运行：python run_parameter_study.py
输出：
  outputs/reports/parameter_study.csv
  outputs/reports/figures/param_{image}.png
"""
import csv
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")          # 不弹窗口，直接存图
import matplotlib.pyplot as plt
import numpy as np

from src.distance_skeleton import distance_skeleton
from src.metrics import skeleton_metrics
from src.preprocessing import fill_small_holes

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "data" / "input"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

SIGMAS = [0.5, 1.0, 1.5, 2.0, 2.5]
TOLERANCES = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
TARGETS = ["ellipse", "cross", "nail_body"]

FIELDS = ["image", "sigma", "tolerance", "time_ms",
          "skeleton_pixels", "endpoints", "branch_points", "branch_pixels",
          "components", "isolated_points", "skeleton_holes", "object_holes",
          "max_radius", "mean_radius"]


def load_target(name):
    if name == "nail_body":
        path = INPUT_DIR / "real" / "nail_body.png"
    else:
        path = INPUT_DIR / "synthetic" / f"{name}.png"
    binary = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if binary is None:
        raise ValueError(f"读取失败:{path}")
    return fill_small_holes(binary)


def draw_heatmap(ax, values, title, log_scale=True):
    data = np.array(values, dtype=float)
    data_plot = np.log10(data + 1) if log_scale else data

    im = ax.imshow(data_plot, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(TOLERANCES)))
    ax.set_xticklabels([str(t) for t in TOLERANCES])
    ax.set_yticks(range(len(SIGMAS)))
    ax.set_yticklabels([str(s) for s in SIGMAS])
    ax.set_xlabel("tolerance")
    ax.set_ylabel("sigma")
    ax.set_title(title)

    vmax = float(data_plot.max()) if data_plot.size else 1.0
    for i in range(len(SIGMAS)):
        for j in range(len(TOLERANCES)):
            ax.text(j, i, str(int(values[i][j])), ha="center", va="center",
                    color="white" if data_plot[i, j] < 0.6 * vmax else "black",
                    fontsize=7)
    return im


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in TARGETS:
        binary = load_target(name)
        print(f"\n[{name}]  单元格格式: 像素/端点/分支/连通分量")
        print("sigma\\tol".ljust(10) + "".join(str(t).rjust(16) for t in TOLERANCES))

        pixels_grid, branch_grid = [], []
        for sigma in SIGMAS:
            px_row, br_row, cells = [], [], []
            for tol in TOLERANCES:
                t0 = time.perf_counter()
                skel = distance_skeleton(binary, sigma=sigma, tolerance=tol)
                t1 = time.perf_counter()
                m = skeleton_metrics(binary, skel)

                px_row.append(m["skeleton_pixels"])
                br_row.append(m["branch_points"])
                cells.append(f"{m['skeleton_pixels']}/{m['endpoints']}/"
                             f"{m['branch_points']}/{m['components']}")

                row = {"image": name, "sigma": sigma, "tolerance": tol,
                       "time_ms": round((t1 - t0) * 1000, 2)}
                row.update(m)
                rows.append(row)

            pixels_grid.append(px_row)
            branch_grid.append(br_row)
            print(str(sigma).ljust(10) + "".join(c.rjust(16) for c in cells))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        im1 = draw_heatmap(axes[0], pixels_grid, f"{name}: skeleton pixels")
        fig.colorbar(im1, ax=axes[0], label="log10(px + 1)")
        im2 = draw_heatmap(axes[1], branch_grid, f"{name}: branch points", log_scale=False)
        fig.colorbar(im2, ax=axes[1])
        fig.tight_layout()

        fig_path = FIGURE_DIR / f"param_{name}.png"
        fig.savefig(fig_path, dpi=110)
        plt.close(fig)
        print("  已保存图:", fig_path)

    csv_path = REPORT_DIR / "parameter_study.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n共 {len(rows)} 行，已保存: {csv_path}")


if __name__ == "__main__":
    main()