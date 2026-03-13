# Contributing

Thank you for your interest in contributing to this project!

## How to Contribute

### Report Issues
- Found a bug or have a suggestion? [Open an issue](https://github.com/Jonny-English/circuits-zoom-in/issues).

### Submit Changes
1. Fork the repository
2. Create a feature branch: `git checkout -b my-feature`
3. Make your changes
4. Ensure the notebook runs end-to-end: `jupyter nbconvert --execute notebooks/circuits_zoom_in_zh.ipynb`
5. Commit and push: `git push origin my-feature`
6. Open a Pull Request

### Contribution Ideas
- **Translations**: Help translate the notebook into other languages
- **New experiments**: Add sections exploring additional circuits or models
- **Visualization improvements**: Better plots, interactive widgets
- **Dataset upgrades**: Replace CIFAR-10 with higher-resolution datasets (e.g., ImageNet subset)
- **Transformer circuits**: Extend the tutorial to cover circuits in language models

## Code Style
- Keep Chinese variable names in the Chinese notebook (this is a deliberate pedagogical choice)
- Add clear comments explaining each step
- Use `matplotlib` for all plots (no external plotting libraries)

## Questions?
Open an issue and we'll be happy to help.
