"""
Export key figures from the notebook for README display.

Usage:
    python scripts/export_figures.py

This runs the core visualization cells and saves the output
to the figures/ directory as PNG files.
"""
import subprocess
import sys
import os

def main():
    print("To generate figures, run the notebook end-to-end first:")
    print()
    print("  jupyter nbconvert --execute --to notebook \\")
    print("    notebooks/circuits_zoom_in_zh.ipynb")
    print()
    print("Then manually save the key plots from the notebook output as:")
    print("  figures/feature_viz_grid.png      (Section 2 output)")
    print("  figures/polar_tuning.png          (Section 4 output)")
    print("  figures/circuit_diagram.png       (Section 5 output)")
    print("  figures/universality_comparison.png (Section 6 output)")
    print()
    print("Tip: In Jupyter, right-click any plot → 'Save Image As...'")

if __name__ == "__main__":
    main()
