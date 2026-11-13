"""生成合成测试集：python make_dataset.py"""
from pathlib import Path

from src.dataset import generate_all

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "input" / "synthetic"

if __name__ == "__main__":
    saved = generate_all(OUTPUT_DIR)
    print(f"已生成 {len(saved)} 张合成图形:")
    for path in saved:
        print("  ", path.relative_to(PROJECT_ROOT))