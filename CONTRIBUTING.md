# Contributing

Thank you for considering a contribution. This project is small and opinionated, but there is plenty of room for it to grow.

## Ways to contribute

**Report problems.** If something in the notebook is wrong — a misattributed claim, a broken cell, a confusing explanation — [open an issue](https://github.com/Jonny-English/circuits-zoom-in/issues). Even a one-sentence description helps.

**Translate.** The notebook currently exists in Chinese and English. If you'd like to translate it into another language, that would extend its reach in a way that no amount of code improvement can.

**Add experiments.** The most natural extensions are deeper circuit analyses (more layers, more models), higher-resolution dataset validation (ImageNet subsets instead of CIFAR-10), and — the big one — a companion tutorial on circuits in Transformer language models.

**Improve visualizations.** The plots work, but they could be clearer, more interactive, or simply more beautiful.

## How to submit changes

1. Fork the repository
2. Create a feature branch: `git checkout -b my-feature`
3. Make your changes
4. Verify the notebook runs end-to-end: `jupyter nbconvert --execute notebooks/circuits_zoom_in_zh.ipynb`
5. Push and open a Pull Request

## A note on Chinese variable names

Please keep them. In the Chinese notebook, variables are named in Chinese. In the English notebook, the same Chinese names are preserved with English annotations. This is intentional and should not be "fixed." If you are adding new code to the Chinese notebook, follow the same convention.

## Questions

Open an issue. I will do my best to respond.
