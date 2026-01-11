# Project Requirements (requirements_EN.md) ✅

## Overview
This document summarizes the core and optional Python packages required to run, train, and evaluate the HydroMoE codebase (data loading, PBM, MoE model, training and evaluation scripts). It also gives recommended versions, installation snippets (pip/conda), GPU/CUDA notes, quick verification steps, and troubleshooting tips.

---

## Environment
- Recommended Python: **3.9** or **3.10**.
- For GPU training, install a PyTorch build that matches your CUDA version (see https://pytorch.org/).

---

## Core dependencies
These packages are required to run the main code paths (data processing, model forward, training):

- python >= 3.9
- numpy >= 1.23
- pandas >= 1.4
- scikit-learn >= 1.0
- torch >= 2.0 (CPU or GPU build)
- tqdm >= 4.60
- pyyaml >= 6.0 (if YAML configuration is used)

---

## Recommended / Optional dependencies
Improve performance, enable Parquet caching, or provide diagnostic tooling:

- pyarrow >= 6.0 or fastparquet — for Parquet/feather caching and faster I/O
- matplotlib, seaborn — plotting and diagnostics
- cma — CMA-ES implementation (if you need to reproduce PBM parameter optimization)
- tensorboard — training visualization (optional)
- joblib — parallel preprocessing or caching utilities

---

## Development / Static analysis (optional)
- black, isort, flake8, mypy — code formatting and linting

---

## System and CUDA notes
- Choose the PyTorch + CUDA combination appropriate for your GPU and driver. See the official install page: https://pytorch.org/
- Windows-specific: DataLoader `num_workers` should be `0` and `pin_memory=False` unless you know how to configure Windows multi-process DataLoader safely (the code already defaults to safe Windows settings).

---

## Installation examples
(a) pip (CPU-only example)

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows PowerShell
pip install --upgrade pip
pip install numpy pandas scikit-learn tqdm pyyaml matplotlib seaborn
pip install torch --index-url https://download.pytorch.org/whl/cpu
# optional: pip install pyarrow cma tensorboard joblib
```

(b) conda (recommended for GPU / Windows)

```bash
conda create -n hydromoe python=3.9 -y
conda activate hydromoe
conda install numpy pandas scikit-learn tqdm pyyaml matplotlib seaborn -c conda-forge
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch  # choose the cudatoolkit matching your driver
conda install pyarrow -c conda-forge  # optional
pip install cma tensorboard joblib  # optional
```

---

## Quick verification (smoke tests) ✅
- Run built-in module tests (these files include small self-tests at bottom):

```bash
python MoE_pbm.py
python MoE_hybrid_model.py
```

- If you have data and want to test data loading:

```bash
python MoE_data_loader.py  # or run warmup_data_loading from MoE_data_loader.py
```

- Evaluate mode (if a saved model exists):

```bash
set EVAL_ONLY=1
python MoE_main_enhanced.py
```

If imports succeed and the scripts print test output, the environment is correctly configured.

---

## Troubleshooting
- Parquet errors → ensure `pyarrow` or `fastparquet` is installed.
- DataLoader errors on Windows → set `num_workers=0` and `pin_memory=False`.
- GPU not detected or CUDA mismatch → reinstall PyTorch with a compatible `cudatoolkit`.

---

## Next steps
If you want, I can:
1. Generate a pinned `requirements.txt` with exact versions (for reproducibility), or
2. Create an `environment.yml` (Conda) that includes a CUDA-aware PyTorch line.

Reply with your preference (pinned vs. loose versions, pip vs. conda) and I will add the file to the repository.
