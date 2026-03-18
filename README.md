# Circuits: Zoom In — A Hands-On Tutorial

[**中文版 README**](README_zh.md) ·
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jonny-English/circuits-zoom-in/blob/main/notebooks/circuits_zoom_in_en.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

> *"If you look carefully enough, you'll find that every neuron is trying to tell you something."*
> — Chris Olah

<p align="center">
  <img src="figures/feature_viz_grid.png" width="45%" alt="Feature visualization grid"/>
  &nbsp;&nbsp;
  <img src="figures/polar_tuning.png" width="45%" alt="Orientation tuning polar plots"/>
</p>
<p align="center">
  <em>Left: "ideal patterns" for 8 neurons — each image is what a neuron most wants to see.</em>
  &nbsp;&nbsp;
  <em>Right: orientation tuning of curve detectors — each neuron has a preferred direction.</em>
</p>

<p align="center">
  <img src="figures/circuit_diagram.png" width="45%" alt="Circuit: edge detectors to curve detector"/>
  &nbsp;&nbsp;
  <img src="figures/universality_comparison.png" width="45%" alt="Universality: InceptionV1 vs ResNet-18"/>
</p>
<p align="center">
  <em>Left: a complete "circuit" — edge detectors combine to compute a curve detector.</em>
  &nbsp;&nbsp;
  <em>Right: two different architectures, trained independently, learn similar visual "vocabulary".</em>
</p>

---

## What this is about

In 2020, Chris Olah and his collaborators at OpenAI published a paper that changed how many of us think about neural networks. The central argument was deceptively simple: if you look carefully enough at the weights and activations inside a vision model, you will find that individual neurons detect meaningful features, that those neurons wire together into interpretable circuits, and that the same features and circuits appear across entirely different architectures. They called the paper *[Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/)*.

This repository is my attempt to reproduce those core experiments from scratch, and to do so in a way that is accessible to readers who think and learn in Chinese. Every variable in the code is named in Chinese characters. Every explanation is written first in Chinese, then translated into English. The result is a bilingual tutorial that covers five experiments from the original paper, plus a section on the limitations and open questions that the paper leaves behind.

## What you will learn, and how

The notebook walks through six experiments, each building on the last:

| Experiment | What you will see |
|-----------|------------------|
| **§2 Feature Visualization** | Starting from random noise, we use gradient ascent to maximize a neuron's activation — asking it: *what are you looking for?* The answers are strikingly recognizable curves, edges, and textures. |
| **§3 Dataset Validation** | We search CIFAR-10 for real photos that most excite the same neuron. When they visually match the synthetic image, we gain confidence the neuron detects a genuine visual pattern, not an optimization artifact. |
| **§4 Orientation Tuning** | We probe the neuron with synthetic arcs at 36 orientations and plot its response as a polar diagram. The result echoes neuroscience directly: InceptionV1's curve detectors show direction selectivity remarkably similar to V1 simple cells in the primate visual cortex. |
| **§5 Circuit Analysis** | We read the weight matrices to trace how a curve detector is *computed* as a weighted combination of upstream edge detectors. Curves aren't detected by magic — they are built from edges through learned weights. |
| **§6 Universality** | We repeat the experiment on ResNet-18 — a completely different architecture, trained independently. When networks that share nothing but training data develop the same features, those features are not accidents but the natural vocabulary of vision. |
| **§7 Limitations** | An honest discussion of what this tutorial does *not* show: polysemanticity, nonlinear interactions, and the gap between vision circuits and Transformer circuits — signposts for your next steps. |

## Why this project exists

There is a growing community of Chinese-speaking researchers and students who want to understand mechanistic interpretability — the subfield of AI safety concerned with reverse-engineering what neural networks have actually learned. The foundational papers are all in English, the code comments are in English, and the variable names are in English. For a native Chinese speaker, learning the concepts means simultaneously navigating a foreign language and a foreign set of abstractions.

This tutorial takes a different approach. The code uses Chinese variable names — `形状记录` instead of `shape_record`, `钩子列表` instead of `hook_list`, `曲线探测器` instead of `curve_detector`. This is not decoration. When you read `shape_record`, your eyes may glide over it as a familiar programming token without engaging with the concept. When you read `形状记录`, you are forced to think about what a "shape record" actually is. The Chinese names are a pedagogical device: they slow you down in exactly the right way.

The English version of the notebook preserves the Chinese variable names but adds English annotations alongside each one, so non-Chinese speakers can follow along too.

Beyond the language, the tutorial is designed for accessibility in a more practical sense: it runs entirely on CPU, requires no GPU, works in Google Colab with a single click, and completes in about fifteen minutes on a laptop. The barrier to entry is as close to zero as I could make it.

## Quick start

```bash
git clone https://github.com/Jonny-English/circuits-zoom-in.git
cd circuits-zoom-in
pip install -r requirements.txt

# Pick your language
jupyter notebook notebooks/circuits_zoom_in_en.ipynb  # English
jupyter notebook notebooks/circuits_zoom_in_zh.ipynb  # 中文
```

Or click the **Open in Colab** badge at the top of this page.

## Project structure

```
circuits-zoom-in/
├── notebooks/
│   ├── circuits_zoom_in_zh.ipynb   # Chinese version (中文版)
│   └── circuits_zoom_in_en.ipynb   # English version
├── utils/                         # Shared utilities (font config, visualization)
├── figures/                        # Pre-rendered figures for this README
├── scripts/                        # Figure generation and utilities
├── requirements.txt
├── pyproject.toml
├── CITATION.cff
├── CONTRIBUTING.md
└── LICENSE
```

## Citation

If you find this tutorial useful in your work, you are welcome to cite it:

```bibtex
@software{circuits_zoom_in_tutorial,
  title  = {Circuits: Zoom In — A Hands-On Tutorial},
  author = {Jonny-English},
  year   = {2026},
  url    = {https://github.com/Jonny-English/circuits-zoom-in},
  license = {MIT}
}
```

## Acknowledgments

This tutorial would not exist without the original [Zoom In](https://distill.pub/2020/circuits/zoom-in/) paper by Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. The [lucent](https://github.com/greentfrapp/lucent) library, maintained by the open-source community, made feature visualization in PyTorch straightforward. And the [Distill](https://distill.pub/) journal — now inactive, but still deeply influential — set the standard for what clear, honest, and visually rich scientific communication can look like.

## License

[MIT](LICENSE)
