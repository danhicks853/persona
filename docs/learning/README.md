# Learning Guide - Deep Principles

**Purpose:** Understand the *why* and *how* behind every technique, not just the *what*.

**Philosophy:** No cargo-culting. Every tool, every technique has scientific principles, engineering trade-offs, and mathematical foundations.

---

## Directory Structure

Organized by theme for intuitive navigation:

```
learning/
├── foundations/     # Core concepts and fundamentals
├── tools/          # Software, frameworks, and packages
├── methods/        # Training techniques and approaches
├── architecture/   # Model architecture and design
└── advanced/       # Optimization and deployment
```

---

## 📚 Foundations - Core Concepts

**Understanding the fundamental building blocks**

### ✅ Hardware Fundamentals
`foundations/01_hardware_fundamentals.md`
- What IS a CUDA core? Tensor core? VRAM?
- Why GPUs for ML? Memory hierarchy explained
- Your RTX 2000 Ada specs and capabilities

### ✅ Python Environments  
`foundations/02_python_environments.md`
- What IS Anaconda? Miniconda? Conda?
- Conda environments vs venv - what's the difference?
- What did we install and why?

### ✅ Neural Network Basics
`foundations/04_neural_network_basics.md`
- What ARE weights? Biases? Neurons?
- How does training actually work?
- Gradients, loss functions, automatic learning
- The machine learns WITHOUT human intervention - how?

---

## 🛠️ Tools - Software & Frameworks

**Understanding your development environment**

### ✅ PyTorch & Frameworks
`tools/01_pytorch_and_frameworks.md`
- What IS PyTorch? Tensors? Autograd?
- Computational graphs and automatic differentiation
- What IS Unsloth and why 2-5x faster?
- Flash Attention and CUDA optimizations

### ✅ Package Reference
`tools/02_package_reference.md` 
- Complete reference for all 80+ packages installed
- What each package does and why it's needed
- Version compatibility notes
- Organized by category (deep learning, GPU, utilities, etc.)

---

## ⚙️ Methods - Training Techniques

**Coming soon - written as we encounter them in Phase 0a+**

### 🚧 Quantization
`methods/01_quantization.md`
- How does 4-bit work? What IS NF4?
- BitsAndBytes internals
- VRAM math: 26GB → 4-7GB
- Trade-offs: memory vs speed vs quality

### 🚧 Fine-Tuning & LoRA
`methods/02_fine_tuning.md`
- LoRA mathematics - why low-rank works
- QLoRA: quantization + LoRA
- Target modules and rank selection
- Adapter efficiency explained

### 🚧 Training Deep Dive
`methods/03_training.md`
- Optimizers: AdamW, SGD, how they differ
- Loss functions for language modeling
- Learning rates, schedulers, warmup
- Gradient accumulation and checkpointing

---

## 🏗️ Architecture - Model Design

**Coming soon - transformer architecture and attention**

### 🚧 Transformers
`architecture/01_transformers.md`
- Attention mechanism - the core innovation
- Self-attention mathematics
- Multi-head attention - why parallel?
- Architecture: encoder-decoder, decoder-only

### 🚧 Context & Attention
`architecture/02_context_attention.md`
- How attention scales O(n²)
- RoPE (Rotary Position Embeddings)
- Context windows: 8K vs 128K
- KV caching for inference

---

## 🚀 Advanced - Optimization & Deployment

**Coming soon - production and efficiency**

### 🚧 Memory Management
`advanced/01_memory.md`
- VRAM management strategies
- Activation checkpointing trade-offs
- Mixed precision (FP32, FP16, BF16)
- Gradient accumulation math

### 🚧 Compression
`advanced/02_compression.md`
- Summarization techniques
- Vector embeddings and retrieval
- Context compression strategies
- Our neurosymbolic approach

### 🚧 Evaluation
`advanced/03_evaluation.md`
- Perplexity - what does it mean?
- BLEU, ROUGE metrics
- Human evaluation strategies
- When metrics lie

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

### **Foundations** (3/3 complete) ✅
- ✅ Hardware fundamentals
- ✅ Python environments
- ✅ Neural network basics

### **Tools** (2/2 complete) ✅
- ✅ PyTorch & frameworks
- ✅ Package reference (all 80+ packages documented!)

### **Methods** (0/3) 🚧
- 🚧 Quantization (coming in Phase 0a)
- 🚧 Fine-tuning & LoRA
- 🚧 Training deep dive

### **Architecture** (0/2) 🚧
- 🚧 Transformers
- 🚧 Context & attention

### **Advanced** (0/3) 🚧
- 🚧 Memory management
- 🚧 Compression
- 🚧 Evaluation

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
