# CUDAgent Setup Guide

This guide will help you set up your Python environment for CUDAgent using Miniconda, install all necessary dependencies, and verify your CUDA and PyTorch installation.

---

## 1. Install Miniconda (if not already installed)

Download and install Miniconda from: https://docs.conda.io/en/latest/miniconda.html

---

## 2. Create a New Conda Environment

Open your terminal and run:

```bash
conda create -n cudagent python=3.9 -y
conda activate cudagent
```

---

## 3. Install CUDA Toolkit (if not already installed)

> **Note:** Make sure your NVIDIA drivers are up to date and compatible with your CUDA version.

```bash
# For CUDA 11.8 (recommended for PyTorch >=1.12)
conda install -c nvidia cuda-toolkit=11.8
```

---

## 4. Install Python Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
# For development:
pip install -r requirements-dev.txt
```

---

## 5. Verify CUDA and PyTorch Installation

Run the following Python commands to check if CUDA is available:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA version:', torch.version.cuda)"
```

You should see:

```
CUDA available: True
CUDA version: 11.8
```

---

## 6. (Optional) Jupyter Notebook Setup

If you want to use Jupyter Notebooks:

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name cudagent --display-name "CUDAgent"
```

---

## 7. Troubleshooting

- Ensure your GPU drivers are up to date.
- Check your CUDA version with `nvcc --version`.
- **NumPy Compatibility**: If you encounter NumPy-related errors, ensure you have NumPy < 2.0 installed: `pip install "numpy<2"`
- If you encounter issues, see the [PyTorch CUDA Troubleshooting Guide](https://pytorch.org/docs/stable/notes/cuda.html).

---

## 8. Deactivating the Environment

When you're done working:

```bash
conda deactivate
```

---

## 9. Removing the Environment (if needed)

```bash
conda remove -n cudagent --all
```

---

You're now ready to start developing with CUDAgent! 🚀 