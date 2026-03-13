# Circuits: Zoom In — A Hands-On Tutorial

[**中文版 README**](README_zh.md) ·
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jonny-English/circuits-zoom-in/blob/main/notebooks/circuits_zoom_in_en.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> A **bilingual (Chinese/English)** educational notebook reproducing the core experiments from [Olah et al. 2020, *Zoom In: An Introduction to Circuits*](https://distill.pub/2020/circuits/zoom-in/).

<p align="center">
  <img src="figures/feature_viz_grid.png" width="45%" alt="Feature visualization grid"/>
  &nbsp;&nbsp;
  <img src="figures/polar_tuning.png" width="45%" alt="Orientation tuning polar plots"/>
</p>
<p align="center">
  <img src="figures/circuit_diagram.png" width="45%" alt="Circuit: edge detectors → curve detector"/>
  &nbsp;&nbsp;
  <img src="figures/universality_comparison.png" width="45%" alt="Universality: InceptionV1 vs ResNet-18"/>
</p>

## What You Will Learn

| Section | Topic | Method |
|---------|-------|--------|
| §2 | Feature Visualization | Activation maximization with [lucent](https://github.com/greentfrapp/lucent) |
| §3 | Dataset Validation | Finding highest-activating real images (CIFAR-10) |
| §4 | Orientation Tuning | Measuring direction selectivity of curve detectors |
| §5 | Circuit Analysis | Tracing how curve detectors are computed from upstream edge detectors via weights |
| §6 | Universality | Cross-architecture comparison (InceptionV1 vs ResNet-18) |
| §7 | Limitations | Polysemanticity, nonlinear interactions, and connections to current research |

## Why This Project

- **Accessibility**: The first comprehensive **Chinese-language** reproduction of the foundational Circuits paper, serving the world's second-largest ML research community
- **Pedagogical innovation**: Uses **Chinese variable names** as a deliberate teaching device — forcing conceptual engagement rather than rote code-copying
- **Zero barrier**: Runs on **CPU** (no GPU required), works in **Google Colab**, takes ~15 minutes end-to-end on a laptop
- **Interdisciplinary**: The orientation tuning experiment (§4) explicitly connects to **visual neuroscience** (V1 simple cell direction selectivity)

## Quick Start

```bash
# Clone
git clone https://github.com/Jonny-English/circuits-zoom-in.git
cd circuits-zoom-in

# Install dependencies
pip install -r requirements.txt

# Run (choose your language)
jupyter notebook notebooks/circuits_zoom_in_en.ipynb  # English
jupyter notebook notebooks/circuits_zoom_in_zh.ipynb  # 中文
```

Or just click the **Open in Colab** badge above — no local setup needed.

## Project Structure

```
circuits-zoom-in/
├── notebooks/
│   ├── circuits_zoom_in_zh.ipynb   # 中文版 (Chinese)
│   └── circuits_zoom_in_en.ipynb   # English version
├── figures/                        # Pre-rendered figures for README
├── scripts/                        # Utility scripts
├── requirements.txt
├── pyproject.toml
├── CITATION.cff
├── CONTRIBUTING.md
└── LICENSE
```

## About the Chinese Variable Names

This project deliberately uses Chinese identifiers (e.g., `形状记录` instead of `shape_record`, `钩子列表` instead of `hook_list`). This is not a quirk — it's a pedagogical choice:

1. **Forces conceptual understanding**: Readers must understand *what* a variable represents, not just pattern-match familiar English tokens
2. **Native-language learning**: Chinese-speaking students engage with ML concepts in their own language
3. **Fully annotated**: The English notebook keeps Chinese names but adds English comments explaining each one

## Citation

```bibtex
@software{circuits_zoom_in_tutorial,
  title = {Circuits: Zoom In — A Hands-On Tutorial},
  author = {Jonny-English},
  year = {2026},
  url = {https://github.com/Jonny-English/circuits-zoom-in},
  license = {MIT}
}
```

## Acknowledgments

- [Chris Olah](https://colah.github.io/) et al. for the original [Zoom In](https://distill.pub/2020/circuits/zoom-in/) paper
- The [lucent](https://github.com/greentfrapp/lucent) library maintainers for making feature visualization accessible in PyTorch
- The [Distill](https://distill.pub/) journal for pioneering clear, interactive ML communication

## License

[MIT](LICENSE)
