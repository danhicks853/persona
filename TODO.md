# TODO - Phase 0a Progress

**Last Updated:** 2025-11-08, 9:45pm  
**Current Phase:** Phase 0a - Toy Project (Day 1)

---

## ✅ Completed

### **Environment Setup**
- [x] Install Miniconda
- [x] Create `unsloth_env` with Python 3.11
- [x] Install PyTorch 2.5.1 + CUDA 12.4
- [x] Install Unsloth + dependencies (80+ packages)
- [x] Fix version compatibility issues (PyTorch 2.6.0 → 2.5.1, removed torchao)
- [x] Verify GPU detection (RTX 2000 Ada, 16GB VRAM)
- [x] Test Unsloth import and functionality

### **Documentation**
- [x] Create `docs/learning/` with foundations, tools, methods structure
- [x] Write 4 foundation modules (hardware, environments, PyTorch, neural networks)
- [x] Document all 80+ packages in `tools/02_package_reference.md`
- [x] Organize learning directory by theme
- [x] Create testing directory structure

### **Repository Organization**
- [x] Create `tests/` directory structure
- [x] Move `test_gpu.py` to `tests/environment/`
- [x] Create `tests/unsloth/test_basic.py`
- [x] Add `tests/README.md`

---

## 🎯 Current Focus: Phase 0a - Day 1

### **Immediate Next Steps**

#### **1. Model Switch & Verification (10-15 minutes)**
- [x] Research Qwen3-1.7B (Chris's recommendation)
- [ ] Clean up old Qwen2.5 model cache to save disk space
- [ ] Update test to use `unsloth/Qwen3-1.7B`
- [ ] Test loading Qwen3-1.7B with 4-bit quantization
- [ ] Verify VRAM usage (~2-4GB expected)
- [ ] Test thinking mode vs non-thinking mode
- [ ] Run one inference to confirm everything works
- [ ] Document results

#### **2. Data Collection (2-3 hours)**
- [x] Create `data/phase0a/` directory structure
- [x] Create data collection template
- [x] Update data collection guide with reasoning examples
- [x] Started reasoning examples collection (11/30 complete - 37%)
- [ ] Collect remaining 19 reasoning examples
- [ ] Collect 15 style examples (direct responses - NON-REASONING)
- [ ] Collect 15 facts examples (info about you - NON-REASONING)
- [ ] Format all as JSONL with <thinking> tags for reasoning examples
- [ ] Split: 50 train, 10 test

**Progress:** 11 reasoning examples captured with high-quality psychology data

---

## 📅 Phase 0a - Remaining Schedule

### **Day 2: Training + Testing** (Tomorrow)
- [ ] Create `scripts/phase0a/train.py`
- [ ] Train Qwen3-1.7B on 50 examples (~100 steps)
- [ ] Monitor VRAM and training time
- [ ] Save checkpoint to `models/checkpoints/phase0a/`
- [ ] Test inference on 10 held-out examples
- [ ] Manual quality review
- [ ] Document training metrics

### **Day 3: Compression + Decision** (Day after)
- [ ] Create `scripts/phase0a/test_compression.py`
- [ ] Implement simple context compression test
- [ ] Test multi-turn conversation compression
- [ ] Measure token reduction vs quality
- [ ] Write `docs/phases/phase0a_results.md`
- [ ] **Decision:** Continue with Qwen, adjust parameters, or pivot?

---

## 🚧 Blocked / Waiting

*None currently - ready to proceed!*

---

## 📝 Notes

### **Version Compatibility**
- **PyTorch 2.5.1** is the sweet spot for Windows + Unsloth
- **torchao removed** - incompatible with PyTorch 2.5.1
- **xformers 0.0.29** - fallback to PyTorch attention (working fine)

### **Environment**
- Environment: `unsloth_env` (Python 3.11)
- Activation: `conda activate unsloth_env`
- All 80+ packages documented in `docs/learning/tools/02_package_reference.md`

### **Testing**
- GPU test: `python tests/environment/test_gpu.py`
- Unsloth test: `python tests/unsloth/test_basic.py`

---

## 🔮 Future (Post Phase 0a)

### **Phase 1: Full Build** (if Phase 0a succeeds)
- Implement full 3-tier architecture
- Knowledge graph integration
- Context compression pipeline
- Multi-turn conversation handling

### **Learning Modules to Write**
- Quantization (methods/)
- Fine-tuning & LoRA (methods/)
- Training deep dive (methods/)
- Transformers (architecture/)
- Context & attention (architecture/)
- Memory management (advanced/)
- Compression (advanced/)
- Evaluation (advanced/)

---

## ⚡ Quick Commands

```bash
# Activate environment
conda activate unsloth_env

# Run tests
python tests/environment/test_gpu.py
python tests/unsloth/test_basic.py

# Check GPU
nvidia-smi

# Check installed packages
pip list | grep -E "(torch|unsloth|transformers|peft|trl)"
```

---

**Next Action:** Create model loading test to verify Qwen-1.5B can be loaded with 4-bit quantization.
