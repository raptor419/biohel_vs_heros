# BioHEL vs HEROS Comparison

Comparison of BioHEL and HEROS rule-learning algorithms on benchmark datasets.

## Datasets

- MUX6, MUX11, MUX20 (Multiplexer problems)
- GAM_A, GAM_C, GAM_E (GAMETES epistatic problems)

## Requirements

- Python 3.8+
- Docker Desktop
- pandas, numpy, matplotlib, seaborn, scikit-learn, openpyxl

## Usage

1. Start Docker Desktop
2. Run the notebook: `biohel_heros_comparison.ipynb`

The notebook will:
- Train HEROS on all datasets
- Build and run BioHEL via Docker
- Generate comparison results and plots
