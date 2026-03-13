# Circuits: Zoom In — 动手教程

[**English README**](README.md) ·
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jonny-English/circuits-zoom-in/blob/main/notebooks/circuits_zoom_in_zh.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> 一个**中英双语**教学 Notebook，复现 [Olah et al. 2020《Zoom In: An Introduction to Circuits》](https://distill.pub/2020/circuits/zoom-in/) 的核心实验。

<p align="center">
  <img src="figures/feature_viz_grid.png" width="45%" alt="特征可视化网格"/>
  &nbsp;&nbsp;
  <img src="figures/polar_tuning.png" width="45%" alt="方向调谐极坐标图"/>
</p>
<p align="center">
  <img src="figures/circuit_diagram.png" width="45%" alt="电路：边缘探测器 → 曲线探测器"/>
  &nbsp;&nbsp;
  <img src="figures/universality_comparison.png" width="45%" alt="普遍性：InceptionV1 vs ResNet-18"/>
</p>

## 你将学到什么

| 章节 | 主题 | 方法 |
|------|------|------|
| §2 | 特征可视化 | 使用 [lucent](https://github.com/greentfrapp/lucent) 做激活最大化 |
| §3 | 数据集验证 | 在真实图片（CIFAR-10）中找激活最高的样本 |
| §4 | 方向调谐 | 测量曲线探测器的方向选择性 |
| §5 | 电路分析 | 通过权重追踪曲线探测器如何由上游边缘探测器计算而来 |
| §6 | 普遍性 | 跨架构对比（InceptionV1 vs ResNet-18） |
| §7 | 局限性 | 多义性、非线性交互，以及与前沿研究的联系 |

## 为什么做这个项目

- **填补空白**：首个全面复现 Circuits 论文的**中文**教程，服务全球第二大 ML 研究社区
- **教学创新**：刻意使用**中文变量名**——强制概念理解，而非复制粘贴英文代码
- **零门槛**：**CPU 可运行**（无需 GPU），支持 **Google Colab**，笔记本电脑约 15 分钟跑完
- **跨学科**：方向调谐实验（§4）明确连接**视觉神经科学**（V1 简单细胞的方向选择性）

## 快速开始

```bash
# 克隆
git clone https://github.com/Jonny-English/circuits-zoom-in.git
cd circuits-zoom-in

# 安装依赖
pip install -r requirements.txt

# 运行（选择你的语言）
jupyter notebook notebooks/circuits_zoom_in_zh.ipynb  # 中文
jupyter notebook notebooks/circuits_zoom_in_en.ipynb  # English
```

或者直接点击上方 **Open in Colab** 徽章——无需本地配置。

## 项目结构

```
circuits-zoom-in/
├── notebooks/
│   ├── circuits_zoom_in_zh.ipynb   # 中文版
│   └── circuits_zoom_in_en.ipynb   # English version
├── figures/                        # README 展示用的预渲染图片
├── scripts/                        # 工具脚本
├── requirements.txt
├── pyproject.toml
├── CITATION.cff
├── CONTRIBUTING.md
└── LICENSE
```

## 引用

```bibtex
@software{circuits_zoom_in_tutorial,
  title = {Circuits: Zoom In — A Hands-On Tutorial},
  author = {Jonny-English},
  year = {2026},
  url = {https://github.com/Jonny-English/circuits-zoom-in},
  license = {MIT}
}
```

## 致谢

- [Chris Olah](https://colah.github.io/) 等人的原始论文 [Zoom In](https://distill.pub/2020/circuits/zoom-in/)
- [lucent](https://github.com/greentfrapp/lucent) 库的维护者们
- [Distill](https://distill.pub/) 期刊对清晰、交互式 ML 传播的开创性贡献

## 许可证

[MIT](LICENSE)
