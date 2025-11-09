# Learning Guide - Deep Principles

**Purpose:** Understand the *why* and *how* behind every technique, not just the *what*.

**Philosophy:** No cargo-culting. Every tool, every technique has scientific principles, engineering trade-offs, and mathematical foundations.

---

## Structure

This is your college-level ML course, built as you go:

### **Module 1: Hardware** ✅ 
`01_hardware_fundamentals.md` - What IS a CUDA core? VRAM? Why GPUs for ML?

### **Module 2: Neural Networks** (Coming soon)
`02_neural_networks.md` - Mathematics of neural networks, backpropagation, gradient descent

### **Module 3: Transformers** (Coming soon)
`03_transformers.md` - Attention mechanism, architecture, why transformers work

### **Module 4: Quantization** (Coming soon)
`04_quantization.md` - How does 4-bit work? NF4? BitsAndBytes internals

### **Module 5: Fine-Tuning** (Coming soon)
`05_fine_tuning.md` - LoRA mathematics, QLoRA, why low-rank works

### **Module 6: Training** (Coming soon)
`06_training.md` - Optimizers, loss functions, learning rates, gradient accumulation

### **Module 7: Memory** (Coming soon)
`07_memory.md` - VRAM management, activation checkpointing, mixed precision

### **Module 8: Context** (Coming soon)
`08_context_attention.md` - How attention scales, RoPE, context windows

### **Module 9: Compression** (Coming soon)
`09_compression.md` - Summarization, retrieval, vector embeddings

### **Module 10: Evaluation** (Coming soon)
`10_evaluation.md` - Perplexity, BLEU, human eval, what metrics actually mean

---

## How to Use

**As you encounter concepts:**
1. See a term you don't understand? Check the relevant module
2. Each module explains the scientific principle, engineering implementation, and mathematics
3. Real examples from this project throughout

**Progressive depth:**
- **Surface:** What it is and what it does
- **Middle:** How it works and why
- **Deep:** Mathematics and implementation details

**You'll build intuition AND understanding.**

---

## Current Status

✅ **Module 1 complete** - Hardware fundamentals (CUDA, VRAM, GPUs)
🚧 **Modules 2-10** - Will be written as we encounter concepts in Phase 0a+

**This grows with the project** - every time you ask "why?", we document the answer here.

---

## Key Principles

### **1. First Principles Thinking**
Don't memorize recipes. Understand why techniques work from fundamentals.

### **2. Mathematics WITH Intuition**
Equations explained in plain language, then formalized.

### **3. Engineering Trade-offs**
Every choice has costs and benefits. We explain both.

### **4. Practical Examples**
Theory grounded in this project's actual implementation.

---

## Example: Understanding Quantization

**Surface level:** "Use 4-bit to save memory"
- Cargo-culting: Just set `load_in_4bit=True`, don't know why

**Our level:**
1. Why full precision doesn't fit (VRAM math)
2. What quantization actually does (reduce bit precision)
3. How NF4 works (optimal levels for normal distribution)
4. Trade-offs (memory vs speed vs quality)
5. When to use what precision (FP32 vs FP16 vs INT8 vs NF4)

**Result:** Can make informed decisions, debug issues, optimize beyond tutorials

---

## Commitment

**You asked for deep understanding - you'll get it.**

Every "what is this?" becomes a section in these guides.

By the end, you'll have:
- ✅ College-level ML understanding
- ✅ Practical implementation experience
- ✅ Ability to read papers and understand them
- ✅ Confidence to tackle new ML problems
- ✅ Knowledge transferable to other domains

**This IS a course. You're just building a project while learning.**

---

*"Give someone a model, they can fine-tune once. Teach someone the principles, they can build anything." - Ancient ML proverb*

*Also: lizard* 🦎
