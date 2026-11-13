# 二值图像骨架提取与算法评估工具

输入二值图像，运行多种骨架提取算法，保存骨架结果，并从**形态**、**拓扑**和**计算效率**三个维度进行量化比较。

## 功能

- 图像读取、灰度化、二值化、孔洞清理
- 距离变换（欧氏/曼哈顿/棋盘，精确/近似）
- 三种骨架算法：Zhang-Suen、Guo-Hall、距离变换中轴近似
- 指标：骨架像素数、端点、分支点、连通分量、孔洞、最大/平均内切圆半径
- 批量处理、CSV 实验记录、并排对比图、参数敏感性热力图
- pytest 回归测试

## 目录结构

```text
skeleton-eval/
├── data/
│   └── input/
│       ├── synthetic/          # 7 个几何真值已知的合成图形
│       └── real/               # 真实图像提取出的物体掩膜
├── outputs/                    # 运行产物（已被 .gitignore 忽略）
│   ├── skeletons/
│   └── reports/
├── src/
│   ├── preprocessing.py        # 读取、二值化、距离变换、连通域、孔洞填充
│   ├── dataset.py              # 合成测试图形生成
│   ├── skeleton.py             # Zhang-Suen + OpenCV 基准
│   ├── distance_skeleton.py    # 距离变换骨架（脊线+连接+细化）
│   └── metrics.py              # 交叉数、端点、分支簇、孔洞等指标
├── tests/                      # pytest 测试
├── main.py                     # 统一入口
├── make_dataset.py             # 生成合成测试集
├── config.json                 # 实验参数
├── requirements.txt
└── README.md

