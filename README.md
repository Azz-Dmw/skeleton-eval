# 二值图像骨架提取与算法评估工具

输入二值图像，运行多种骨架提取算法，保存骨架结果，并从**形态**、**拓扑**和**计算效率**三个维度进行量化比较。

## 功能

- 图像读取、灰度化、二值化、孔洞清理
- 距离变换（欧氏距离，精确算法）
- 三种骨架算法：Zhang-Suen、Guo-Hall、距离变换中轴近似
- 指标：骨架像素数、端点、分支点、连通分量、孔洞、最大/平均内切圆半径
- 批量处理、CSV 实验记录、并排对比图、参数敏感性热力图
- pytest 回归测试（27 个测试）

## 目录结构

```text
skeleton-eval/
├── data/
│   └── input/
│       ├── synthetic/              # 7 个几何真值已知的合成图形
│       └── real/                   # 真实图像提取出的物体掩膜
├── outputs/                        # 运行产物（.gitignore 忽略）
│   ├── skeletons/
│   └── reports/
│       └── figures/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # 读取、二值化、距离变换、连通域、孔洞填充
│   ├── dataset.py                  # 合成测试图形生成
│   ├── skeleton.py                 # Zhang-Suen + OpenCV 基准
│   ├── distance_skeleton.py        # 距离变换骨架（脊线+连接+细化）
│   └── metrics.py                  # 交叉数、端点、分支簇、孔洞等指标
├── tests/
│   ├── test_preprocessing.py
│   ├── test_skeleton.py
│   ├── test_metrics.py
│   └── test_distance_skeleton.py
├── main.py                         # 统一入口（单张/批量）
├── make_dataset.py                 # 生成合成测试集
├── run_thinning.py                 # 细化算法对比实验
├── run_metrics.py                  # ZS/GH 指标统计
├── run_distance_skeleton.py        # DT/ZS/GH 三方对比
├── run_parameter_study.py          # sigma × tolerance 敏感性实验
├── conftest.py                     # pytest 路径配置
├── pytest.ini
├── config.json                     # 实验参数
├── requirements.txt
└── README.md
```

## 环境

- Python 3.12
- 依赖见 `requirements.txt`

```bash
pip install -r requirements.txt
```

注意：必须用 `opencv-contrib-python`，普通 `opencv-python` 不含 `cv2.ximgproc`（细化算法依赖它）。

## 快速开始

```bash
# 1. 生成合成测试集（7 张图）
python make_dataset.py

# 2. 批量运行三种算法，保存骨架和指标 CSV
python main.py --input data/input/synthetic --binary

# 3. 单张真实图像只跑距离变换骨架
python main.py --input data/input/real/nail_body.png --binary --method dt

# 4. 运行回归测试
python -m pytest -q
```

说明：

- `--binary` 表示输入已经是 `{0, 255}` 的二值图，跳过阈值二值化；合成图和 `nail_body.png` 都必须加这个参数。
- 每次运行会覆盖 `outputs/reports/metrics_main.csv`；想保留不同实验，用 `--output` 指定不同目录。
- 参数从 `config.json` 读取（阈值、极性、孔洞面积上限、细化最大迭代、DT 骨架的 sigma/tolerance 等）。

## 算法说明

| 算法 | 类型 | 说明 |
|---|---|---|
| Zhang-Suen | 细化 | 自实现，与 OpenCV 实现逐像素一致（diff = 0） |
| Guo-Hall | 细化 | 调用 `cv2.ximgproc.thinning`，作为基准 |
| 距离变换中轴近似 | 中轴 | 自实现：DT → 方向性非极大值抑制 → 片段连接 → 细化 |

## 指标定义

- **端点**：交叉数 `A(p) = 1`
- **分支点**：交叉数 `A(p) ≥ 3`，相邻分支像素聚成一簇后计数
- **连通分量**：骨架的 8 连通分量数
- **孔洞**：不接触图像边界的背景连通域数
- **最大内切圆半径**：物体距离变换的最大值
- **平均内切圆半径**：骨架像素处距离变换值的均值，衡量骨架是否居中

用交叉数而不是邻域度数，是为了避免 8 连通下"2-2 阶梯"把普通线段像素误判为分支点。

## 实验脚本

| 脚本 | 作用 | 输出 |
|---|---|---|
| `run_thinning.py` | 三种细化对比 | 并排图 + 像素统计 |
| `run_metrics.py` | ZS/GH 指标统计 | `metrics.csv` |
| `run_distance_skeleton.py` | DT/ZS/GH 三方对比 | `metrics_comparison.csv` + 对比图 |
| `run_parameter_study.py` | sigma/tolerance 扫描 | `parameter_study.csv` + 热力图 |

## 实验结论（摘要）

| 图像 | DT 骨架（像素/端点/分支） | Zhang-Suen | Guo-Hall |
|---|---|---|---|
| comb | 458 / 7 / 5 | 463 / 7 / 5 | 454 / 7 / 5 |
| cross | 295 / 4 / 1 | 299 / 4 / 1 | 301 / 4 / 1 |
| ring（1 孔洞） | 456 / 0 / 0 | 476 / 0 / 0 | 420 / 0 / 0 |
| nail_body | 8 / 2 / 0 | 64 / 2 / 0 | 717 / 9 / 7 |

- 三种算法在合成图形上都复现了理论拓扑（梳状 7 端点/5 分支、十字 4/1、圆环 1 孔洞）；
- Guo-Hall 在真实图像上产生大量低内切圆半径的短枝（平均半径 45.9，DT 骨架为 118.3）；
- 距离变换骨架对 `tolerance` 敏感，必须远小于 1，否则整块物体会被判为脊线；
- 一个 1 像素的孔洞就能让 Zhang-Suen 骨架从 64 像素变成 429 像素，预处理必须包含孔洞清理。

## 已知限制

- 离散骨架只是严格中轴的近似：椭圆上理论中轴长 159.5，DT 骨架 92、ZS 骨架 131；
- 脊线片段连接是启发式（最近邻直线），复杂形状可能连错；
- 真实照片需要先做物体分割，本项目提供的是分割后的掩膜；
- 尚未实现 Delta Medial Axis 等论文算法。

## 路线图

- [ ] Delta Medial Axis 等论文算法
- [ ] 真实指甲图像的颜色/区域分割，让 `main.py` 直接处理原始照片
- [ ] pandas 实验数据汇总与统计检验
- [ ] 参数自动选择与鲁棒性改进