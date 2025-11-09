# Package Reference

**Complete guide to every package installed in the `unsloth_env` environment**

This document provides brief descriptions of all packages installed during environment setup, organized by category. Use this as a reference to understand what each component does and why it's needed.

---

## Core Deep Learning Framework

### **PyTorch 2.5.1**
- **What**: Open-source deep learning framework
- **Why**: Foundation for all neural network operations, provides tensors, autograd, GPU acceleration
- **Key Features**: Automatic differentiation, dynamic computation graphs, CUDA support
- **Used For**: Building, training, and running neural networks

### **torchvision 0.20.1**
- **What**: Computer vision utilities for PyTorch
- **Why**: Provides image transformations, datasets, and pre-trained models
- **Used For**: Image preprocessing (not primary focus, but included with PyTorch)

### **torchaudio 2.5.1**
- **What**: Audio processing utilities for PyTorch
- **Why**: Audio transformations and datasets
- **Used For**: Audio tasks (not used in our project, but comes with PyTorch)

---

## GPU Acceleration & Optimization

### **CUDA 12.4**
- **What**: NVIDIA's parallel computing platform
- **Why**: Enables GPU acceleration for training (20-100x faster than CPU)
- **Key Features**: Parallel matrix operations, tensor cores
- **Hardware**: Required for GPU training on NVIDIA cards

### **xformers 0.0.29**
- **What**: Memory-efficient attention mechanisms
- **Why**: Reduces VRAM usage and speeds up transformer attention
- **Key Features**: Flash Attention, memory-efficient operations
- **Used For**: Faster, lower-memory training of large models

### **triton-windows 3.5.0**
- **What**: GPU programming language/compiler for Windows
- **Why**: Enables custom CUDA kernels for optimizations
- **Key Features**: Easier than raw CUDA, used by many optimization libraries
- **Used For**: Backend for Unsloth optimizations

### **bitsandbytes 0.48.2**
- **What**: 8-bit and 4-bit quantization library
- **Why**: Reduces model memory from 26GB → 4-7GB
- **Key Features**: QLoRA, 4-bit NormalFloat (NF4), paged optimizers
- **Used For**: Loading models in 4-bit for training on consumer GPUs

---

## Training Frameworks & Optimization

### **Unsloth 2025.11.2**
- **What**: CUDA-optimized training framework
- **Why**: 2-5x faster training than standard methods
- **Key Features**: Optimized kernels, memory efficiency, Windows support
- **Used For**: Fast fine-tuning with minimal code changes

### **unsloth-zoo 2025.11.3**
- **What**: Unsloth's model zoo and patches
- **Why**: Pre-configured optimizations for popular models
- **Key Features**: Model-specific patches, efficiency improvements
- **Used For**: Automatic optimization of Qwen and other models

### **accelerate 1.11.0**
- **What**: Hugging Face's distributed training library
- **Why**: Simplifies multi-GPU, mixed-precision training
- **Key Features**: Automatic device placement, gradient accumulation
- **Used For**: Training infrastructure (handles GPU operations)

### **trl 0.24.0** (Transformer Reinforcement Learning)
- **What**: Reinforcement learning for language models
- **Why**: Provides SFTTrainer (Supervised Fine-Tuning)
- **Key Features**: RLHF, DPO, PPO, SFT
- **Used For**: Fine-tuning loop and training utilities

### **peft 0.17.1** (Parameter-Efficient Fine-Tuning)
- **What**: LoRA and other efficient fine-tuning methods
- **Why**: Fine-tune only 1-2% of parameters instead of 100%
- **Key Features**: LoRA, QLoRA, Prefix Tuning, Adapters
- **Used For**: Memory-efficient fine-tuning

---

## Model & Data Handling

### **transformers 4.57.1**
- **What**: Hugging Face's transformer model library
- **Why**: Access to 100,000+ pre-trained models
- **Key Features**: Model hub, tokenizers, trainers
- **Used For**: Loading Qwen models, inference

### **datasets 4.3.0**
- **What**: Hugging Face's dataset loading library
- **Why**: Efficient data loading and processing
- **Key Features**: Memory mapping, streaming, caching
- **Used For**: Loading and formatting training data

### **tokenizers 0.22.1**
- **What**: Fast tokenization library (Rust-based)
- **Why**: Converts text → numbers for models
- **Key Features**: BPE, WordPiece, SentencePiece support
- **Used For**: Text preprocessing for Qwen

### **sentencepiece 0.2.1**
- **What**: Unsupervised text tokenizer
- **Why**: Used by many modern LLMs (including Qwen)
- **Key Features**: Subword tokenization, language-agnostic
- **Used For**: Qwen's tokenization

---

## Utilities & Support Libraries

### **huggingface-hub 0.36.0**
- **What**: Client library for Hugging Face Hub
- **Why**: Downloads models and datasets
- **Key Features**: Authentication, caching, versioning
- **Used For**: Downloading Qwen models

### **safetensors 0.6.2**
- **What**: Safe tensor serialization format
- **Why**: Faster, safer model loading than pickle
- **Key Features**: Security, speed, memory efficiency
- **Used For**: Loading model weights

### **protobuf 6.33.0**
- **What**: Google's data serialization format
- **Why**: Used by some model formats
- **Key Features**: Compact, fast, cross-platform
- **Used For**: Model metadata

### **cut-cross-entropy 25.1.1**
- **What**: Optimized cross-entropy loss
- **Why**: Faster loss computation for language modeling
- **Key Features**: Fused operations, memory efficiency
- **Used For**: Training loss calculation

### **hf-transfer 0.1.9**
- **What**: Fast file downloads for Hugging Face
- **Why**: Multi-threaded downloads (faster model downloads)
- **Key Features**: Parallel transfers, resume support
- **Used For**: Downloading large model files

---

## Data Processing

### **numpy 2.3.3**
- **What**: Numerical computing library
- **Why**: Array operations, math functions
- **Key Features**: N-dimensional arrays, linear algebra
- **Used For**: Data preprocessing, numerical operations

### **pandas 2.3.3**
- **What**: Data manipulation library
- **Why**: DataFrames, CSV handling
- **Key Features**: Tabular data, time series
- **Used For**: Data collection and formatting

### **pyarrow 22.0.0**
- **What**: Apache Arrow for Python
- **Why**: Fast columnar data format
- **Key Features**: Zero-copy reads, compression
- **Used For**: Efficient data storage (used by datasets)

### **fsspec 2025.9.0**
- **What**: File system abstraction
- **Why**: Unified interface for local/cloud storage
- **Key Features**: S3, GCS, HTTP support
- **Used For**: Loading data from various sources

---

## Networking & I/O

### **requests 2.32.5**
- **What**: HTTP library for Python
- **Why**: API calls, downloads
- **Used For**: Downloading models and data

### **aiohttp 3.13.2**
- **What**: Async HTTP client/server
- **Why**: Non-blocking downloads
- **Used For**: Concurrent model downloads

### **httpx 0.28.1**
- **What**: Modern HTTP client
- **Why**: Async support, HTTP/2
- **Used For**: Hugging Face Hub API calls

---

## Development & CLI

### **tyro 0.9.35**
- **What**: CLI argument parsing
- **Why**: Easy command-line interfaces
- **Used For**: Unsloth's command-line tools

### **rich 14.2.0**
- **What**: Terminal formatting library
- **Why**: Beautiful console output
- **Key Features**: Progress bars, syntax highlighting
- **Used For**: Training progress displays

### **tqdm 4.67.1**
- **What**: Progress bar library
- **Why**: Shows training/download progress
- **Used For**: Visual feedback during operations

### **colorama 0.4.6**
- **What**: Cross-platform colored terminal text
- **Why**: Color support on Windows
- **Used For**: Colored output in terminal

---

## Type Checking & Validation

### **typeguard 4.4.4**
- **What**: Runtime type checking
- **Why**: Validates type hints at runtime
- **Used For**: Unsloth's type safety

### **typing-extensions 4.15.0**
- **What**: Backported typing features
- **Why**: Modern type hints for older Python
- **Used For**: Type annotations

---

## System & Utilities

### **psutil 7.1.3**
- **What**: System monitoring library
- **Why**: CPU, memory, GPU monitoring
- **Used For**: Resource usage tracking

### **filelock 3.19.1**
- **What**: File locking mechanism
- **Why**: Prevents concurrent writes
- **Used For**: Cache management

### **packaging 25.0**
- **What**: Package version utilities
- **Why**: Version parsing, comparison
- **Used For**: Dependency management

---

## Mathematics & Symbolic

### **sympy 1.13.1**
- **What**: Symbolic mathematics library
- **Why**: Required by PyTorch
- **Used For**: Symbolic computation in PyTorch internals

### **mpmath 1.3.0**
- **What**: Multi-precision arithmetic
- **Why**: High-precision math
- **Used For**: Backend for sympy

---

## Compression & Hashing

### **xxhash 3.6.0**
- **What**: Fast non-cryptographic hash
- **Why**: Checksum for data integrity
- **Used For**: Dataset caching

### **dill 0.4.0**
- **What**: Extended pickle library
- **Why**: Serializes complex Python objects
- **Used For**: Caching data processing

---

## Text Processing

### **regex 2025.11.3**
- **What**: Regular expression library
- **Why**: Pattern matching in text
- **Used For**: Tokenization, text processing

### **pyyaml 6.0.3**
- **What**: YAML parser
- **Why**: Configuration files
- **Used For**: Model configs, training settings

---

## Network & Data Structures

### **certifi 2025.10.5**
- **What**: SSL certificate bundle
- **Why**: HTTPS connections
- **Used For**: Secure downloads

### **charset-normalizer 3.4.4**
- **What**: Character encoding detection
- **Why**: Handle various text encodings
- **Used For**: Text data loading

### **urllib3 2.5.0**
- **What**: HTTP client
- **Why**: Low-level HTTP operations
- **Used For**: Backend for requests

### **multidict 6.7.0**
- **What**: Dictionary with multiple values per key
- **Why**: HTTP headers
- **Used For**: aiohttp backend

---

## Formatting & Rendering

### **markdown-it-py 4.0.0**
- **What**: Markdown parser
- **Why**: Render markdown text
- **Used For**: Rich text in terminals

### **pygments 2.19.2**
- **What**: Syntax highlighting
- **Why**: Code coloring in output
- **Used For**: Rich library rendering

### **pillow 11.3.0**
- **What**: Image processing library
- **Why**: Image loading and manipulation
- **Used For**: torchvision backend

---

## Miscellaneous

### **six 1.17.0**
- **What**: Python 2/3 compatibility
- **Why**: Legacy compatibility
- **Used For**: Old library dependencies

### **python-dateutil 2.9.0**
- **What**: Date/time utilities
- **Why**: Date parsing and manipulation
- **Used For**: pandas backend

### **pytz 2025.2** & **tzdata 2025.2**
- **What**: Timezone databases
- **Why**: Timezone support
- **Used For**: pandas time series

---

## Summary by Importance

### **Critical (Cannot train without)**
1. PyTorch - Neural network framework
2. CUDA - GPU acceleration
3. transformers - Model loading
4. datasets - Data loading
5. bitsandbytes - 4-bit quantization
6. peft - LoRA implementation
7. trl - Training loop

### **High Priority (Major benefits)**
1. Unsloth + unsloth-zoo - 2-5x speed boost
2. xformers - Memory efficiency
3. accelerate - Training infrastructure

### **Supporting (Required dependencies)**
1. tokenizers, sentencepiece - Text processing
2. safetensors - Model loading
3. huggingface-hub - Downloads
4. All utilities and system libraries

### **Optional (Nice to have)**
1. rich, tqdm - Progress displays
2. psutil - Monitoring
3. triton-windows - Advanced optimizations

---

## Version Compatibility Notes

**Why these specific versions?**

- **PyTorch 2.5.1**: Latest stable with Windows Triton support
- **torchao**: REMOVED - incompatible with PyTorch 2.5.1 (needs 2.6.0+)
- **xformers 0.0.29**: Matched to PyTorch 2.5.1
- **CUDA 12.4**: Compatible with PyTorch 2.5.1 + Windows
- **triton-windows 3.5.0**: Windows fork, requires PyTorch ≥ 2.4

**Critical incompatibility avoided:**
- PyTorch 2.6.0 had `torch.int1` that torchao required but broke other things
- PyTorch 2.4.1 had torch._inductor.config issues
- Solution: PyTorch 2.5.1 is the sweet spot for Windows + Unsloth

---

## Environment Activation

```bash
conda activate unsloth_env
```

All these packages are available when the environment is active!
