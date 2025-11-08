# Environment Setup Guide

## Hardware Requirements

### Your System
- **GPU:** RTX ADA 2000 (20GB VRAM) ✅
- **RAM:** 64GB ✅
- **Storage:** ~100GB free for models + data

### What This Can Handle
- **4-bit models:** Up to 13B parameters
- **Training:** 8B models with QLoRA
- **Inference:** 8B models comfortably
- **Small models:** 1B-3B with plenty of headroom

## Software Prerequisites

### Windows Setup (Your System)

1. **Python 3.10 or 3.11** (not 3.12 - some ML libs not compatible yet)
   ```powershell
   # Check current version
   python --version
   
   # If needed, download from python.org
   # Recommended: 3.11.x
   ```

2. **CUDA Toolkit** (for GPU acceleration)
   ```powershell
   # Check if already installed
   nvidia-smi
   
   # If not, download CUDA 12.1 from:
   # https://developer.nvidia.com/cuda-downloads
   ```

3. **Git** (for version control)
   ```powershell
   git --version
   # If not installed, download from git-scm.com
   ```

4. **Git LFS** (for large model files)
   ```powershell
   git lfs install
   ```

## Python Environment

### Option A: Conda (Recommended)
```powershell
# Install Miniconda if needed
# https://docs.conda.io/en/latest/miniconda.html

# Create environment
conda create -n persona python=3.11
conda activate persona

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Verify GPU is detected
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### Option B: venv (Alternative)
```powershell
# Create environment
python -m venv venv
.\venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## Install ML Dependencies

```powershell
# Core ML libraries
pip install transformers accelerate
pip install bitsandbytes  # For 4-bit quantization
pip install peft  # For LoRA adapters
pip install trl  # For training (SFT, DPO)
pip install datasets  # For data handling

# Utilities
pip install sentencepiece protobuf
pip install scipy
pip install safetensors

# Optional but useful
pip install huggingface_hub  # For downloading models
pip install gradio  # For UI (if you want)

# Development tools
pip install jupyter  # For experimentation
pip install matplotlib seaborn  # For visualizations
```

## Verify Installation

Create a test script: `test_setup.py`

```python
import torch
import transformers
import peft
import trl
import bitsandbytes

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Transformers version: {transformers.__version__}")
print(f"PEFT version: {peft.__version__}")
print(f"TRL version: {trl.__version__}")
print("\nAll libraries installed successfully!")
```

Run it:
```powershell
python test_setup.py
```

Expected output:
```
PyTorch version: 2.x.x
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA RTX ADA 2000
VRAM: 20.00 GB
Transformers version: 4.x.x
PEFT version: 0.x.x
TRL version: 0.x.x

All libraries installed successfully!
```

## Directory Structure

The repo expects this structure (auto-created):

```
d:\Github\persona\
├── data\
│   ├── raw\              # Your exported data (gitignored)
│   ├── processed\        # Cleaned datasets
│   │   ├── facts\
│   │   ├── style\
│   │   └── decisions\
│   └── scripts\          # Processing scripts
├── models\
│   ├── base\             # Downloaded base models
│   └── adapters\         # Your trained LoRAs
├── training\             # Training scripts
├── evaluation\           # Test scripts
├── inference\            # Chat interface
└── docs\                 # Documentation
```

Create it:
```powershell
cd d:\Github\persona
mkdir data\raw, data\processed\facts, data\processed\style, data\processed\decisions, data\scripts
mkdir models\base, models\adapters
mkdir training, evaluation, inference
```

## .gitignore Setup

Create `.gitignore`:
```
# Python
__pycache__/
*.py[cod]
*$py.class
venv/
env/

# Data (keep private)
data/raw/**
data/processed/**
*.jsonl
*.json

# Models
models/base/**
models/adapters/**
*.bin
*.safetensors
*.gguf

# Outputs
*.log
*.txt
wandb/
outputs/

# Personal
.env
secrets.json
```

## Environment Variables

Create `.env` file:
```bash
# Hugging Face (for model downloads)
HF_TOKEN=your_token_here  # Optional, only needed for gated models

# API Keys (for tool use later)
ANTHROPIC_API_KEY=your_key_here  # For Claude
OPENAI_API_KEY=your_key_here     # If using GPT

# Paths
MODEL_CACHE_DIR=d:/Github/persona/models/base
DATA_DIR=d:/Github/persona/data
```

## Troubleshooting

### GPU Not Detected
```powershell
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory Errors
- Use 4-bit quantization (already in config)
- Reduce batch size
- Use gradient checkpointing
- Close other GPU applications

### Slow Downloads
- Use `huggingface-cli` with resume support
- Check network connection
- Consider mirrors for large models

### Import Errors
```powershell
# Reinstall with specific versions
pip install transformers==4.36.0
pip install peft==0.7.0
pip install trl==0.7.0
```

## Next Steps

Once setup is complete:
1. ✅ Environment working
2. ✅ GPU detected
3. ✅ Libraries installed
4. → **Proceed to Phase 0** (`docs/phases/phase0.md`)

## Estimated Time

- Fresh install: 1-2 hours
- Already have Python/CUDA: 15-30 minutes
- Just need packages: 5-10 minutes

---

*Setup guide for Windows 11, RTX ADA 2000, 64GB RAM*
