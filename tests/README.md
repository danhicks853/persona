# Tests Directory

**Purpose:** Organized testing structure for environment, framework, and model verification.

---

## Directory Structure

```
tests/
├── environment/       # Environment and GPU tests
│   └── test_gpu.py   # Verify PyTorch CUDA and GPU detection
├── unsloth/          # Unsloth framework tests
│   └── test_basic.py # Verify Unsloth import and basic functionality
├── model_loading/    # Model loading and inference tests
│   └── (future tests for Qwen loading and inference)
└── README.md         # This file
```

---

## Running Tests

### **Environment Tests**
```bash
conda activate unsloth_env
python tests/environment/test_gpu.py
```

### **Unsloth Tests**
```bash
conda activate unsloth_env
python tests/unsloth/test_basic.py
```

---

## Test Categories

### **environment/** - System Setup
- GPU detection and CUDA availability
- PyTorch version and compatibility
- VRAM detection

### **unsloth/** - Framework Verification
- Unsloth import and initialization
- FastLanguageModel availability
- Optimization patches applied correctly

### **model_loading/** - Model Operations
- Qwen model download and loading
- 4-bit quantization verification
- Inference testing
- VRAM usage monitoring

**Run model loading test:**
```bash
conda activate unsloth_env
python tests/model_loading/test_qwen_load.py
```

---

## Adding New Tests

When adding a new test:
1. Create it in the appropriate subdirectory
2. Name it `test_*.py` for clarity
3. Include clear success/failure output
4. Document what it tests in this README

---

## Current Status

✅ **environment/test_gpu.py** - Working
✅ **unsloth/test_basic.py** - Created
🚧 **model_loading/** - Coming in Phase 0a
