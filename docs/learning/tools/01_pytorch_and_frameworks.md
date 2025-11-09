# PyTorch & ML Frameworks

**Why this matters:** Understanding your tools lets you debug issues and optimize performance.

---

## What IS PyTorch?

**Short answer:** A deep learning framework (library for building and training neural networks)

**Long answer:** A Python library that makes it easy to:
1. Build neural networks
2. Train them on GPUs
3. Compute gradients automatically
4. Handle large datasets efficiently

---

## The Problem PyTorch Solves

### **Without a Framework:**

```python
# You'd have to manually implement everything:

# Forward pass (compute output)
def forward(x, weights):
    layer1 = relu(x @ weights[0] + bias[0])
    layer2 = relu(layer1 @ weights[1] + bias[1])
    output = softmax(layer2 @ weights[2] + bias[2])
    return output

# Backward pass (compute gradients) - THE HARD PART
def backward(x, y_true, weights):
    # Manually compute gradient for each weight
    # Chain rule through every layer
    # This gets EXTREMELY complicated for deep networks
    grad_w2 = ...  # 50 lines of calculus
    grad_w1 = ...  # 100 lines of calculus
    grad_w0 = ...  # 150 lines of calculus
    return [grad_w0, grad_w1, grad_w2]

# GPU acceleration - write CUDA kernels yourself
__global__ void matmul_kernel(...) {
    // 200 lines of CUDA code per operation
}

# This is why research was slow before frameworks!
```

### **With PyTorch:**

```python
import torch
import torch.nn as nn

# Build model (PyTorch handles the math)
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax()
)

# Forward pass
output = model(x)

# Backward pass (automatic!)
loss = loss_function(output, y_true)
loss.backward()  # PyTorch computes ALL gradients automatically!

# Move to GPU (one line)
model = model.cuda()

# That's it! PyTorch handles the hard parts.
```

---

## What IS a Tensor?

**The fundamental data structure in PyTorch**

### **Conceptually:**

```
Tensor = Multi-dimensional array of numbers

Scalar (0D tensor): 5
Vector (1D tensor): [1, 2, 3, 4]
Matrix (2D tensor): [[1, 2], [3, 4]]
3D tensor: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
...
ND tensor: Any number of dimensions
```

### **Why not just NumPy arrays?**

**NumPy:**
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = a + b  # [5, 7, 9]

# But:
- Runs on CPU only
- No automatic gradients
- Can't easily move to GPU
```

**PyTorch tensors:**
```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a + b  # tensor([5, 7, 9])

# Advantages:
- Can run on GPU: a.cuda()
- Automatic gradients: requires_grad=True
- Seamless CPU/GPU transfer
- Drop-in NumPy replacement (mostly compatible API)
```

### **Example:**

```python
# Create tensor
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Shape
print(x.shape)  # torch.Size([2, 2])

# Move to GPU
x_gpu = x.cuda()  # Now on GPU
x_cpu = x_gpu.cpu()  # Back to CPU

# Enable gradient tracking
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Compute something
y = (x ** 2).sum()  # y = 1^2 + 2^2 + 3^2 = 14

# Automatic gradient!
y.backward()
print(x.grad)  # tensor([2., 4., 6.]) = [2*1, 2*2, 2*3]
```

---

## How PyTorch Works: Computational Graphs

**The key insight:** PyTorch builds a graph of operations and uses it to compute gradients

### **Forward Pass (Building the Graph):**

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2      # Operation 1: square
z = y + 3       # Operation 2: add 3
w = z * 4       # Operation 3: multiply 4

# PyTorch secretly built this graph:
#
# x (2.0)
#  |
#  | (square)
#  v
# y (4.0)
#  |
#  | (add 3)
#  v
# z (7.0)
#  |
#  | (multiply 4)
#  v
# w (28.0)
```

### **Backward Pass (Using the Graph):**

```python
w.backward()  # Compute gradient of w with respect to x

# PyTorch walks the graph backwards:
#
# ∂w/∂z = 4 (multiply operation)
# ∂z/∂y = 1 (add operation)
# ∂y/∂x = 2*x = 2*2 = 4 (square operation)
#
# Chain rule: ∂w/∂x = (∂w/∂z) * (∂z/∂y) * (∂y/∂x)
#                    = 4 * 1 * 4
#                    = 16

print(x.grad)  # tensor([16.0])
```

**This is called "automatic differentiation" or "autograd"**

---

## PyTorch Components

### **1. torch (Core)**

```python
import torch

# Tensor creation
x = torch.zeros(3, 4)  # 3x4 matrix of zeros
x = torch.randn(2, 3)  # Random normal distribution
x = torch.tensor([1, 2, 3])  # From Python list

# Operations
y = x + 5
y = torch.matmul(a, b)  # Matrix multiplication
y = x.mean()  # Statistics
```

### **2. torch.nn (Neural Network Building Blocks)**

```python
import torch.nn as nn

# Layers
linear = nn.Linear(10, 5)  # Fully connected: 10 inputs → 5 outputs
conv = nn.Conv2d(3, 64, 3)  # Convolution for images
lstm = nn.LSTM(10, 20)  # Recurrent layer for sequences

# Activation functions
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
gelu = nn.GELU()  # Used in transformers

# Loss functions
criterion = nn.CrossEntropyLoss()  # For classification
criterion = nn.MSELoss()  # For regression

# Build a model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),  # Regularization
    nn.Linear(256, 10)
)
```

### **3. torch.optim (Optimization Algorithms)**

```python
import torch.optim as optim

# Optimizers
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Training step
optimizer.zero_grad()  # Clear old gradients
loss.backward()        # Compute gradients
optimizer.step()       # Update weights
```

### **4. torch.utils.data (Data Loading)**

```python
from torch.utils.data import DataLoader, Dataset

# Custom dataset
class MyDataset(Dataset):
    def __len__(self):
        return 1000  # Number of examples
    
    def __getitem__(self, idx):
        # Return one example
        return data[idx], label[idx]

# Data loader (batching, shuffling, parallel loading)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4  # Parallel data loading
)

for batch_data, batch_labels in loader:
    # Train on batch
    ...
```

---

## What We Installed

### **When we ran:**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### **What we got:**

**1. PyTorch (1.21 GB):**
```
pytorch-2.5.1
- Core tensor library
- Autograd engine
- Neural network modules
- Optimizers
- CUDA bindings
```

**2. CUDA libraries (~500 MB):**
```
libcublas, libcublasLt    Matrix operations on GPU
libcudnn                  Neural network primitives (optimized conv, pool, etc.)
libcufft                  Fast Fourier Transform on GPU
libcurand                 Random number generation on GPU
libcusolver               Linear algebra solvers on GPU
libcusparse               Sparse matrix operations
libnvjpeg                 JPEG encoding/decoding on GPU
```

**3. torchvision:**
```
- Image processing utilities
- Pre-trained vision models (ResNet, VGG, etc.)
- Dataset loaders (ImageNet, CIFAR, etc.)
- Transforms (resize, crop, normalize, etc.)

We won't use this (we're doing NLP), but comes in the bundle
```

**4. torchaudio:**
```
- Audio processing utilities
- Transforms (spectrograms, mel-scale, etc.)

We won't use this either (NLP project)
```

**5. NumPy, MKL (Intel Math Kernel Library):**
```
numpy-2.0.1              NumPy arrays (PyTorch can convert to/from)
mkl                      Optimized CPU math operations
mkl_fft                  Fast Fourier Transform (CPU)
mkl_random               Fast random number generation (CPU)
```

### **Total installed: ~2.5 GB**

---

## What IS Unsloth?

**Short answer:** A library that makes fine-tuning large language models 2-5x faster and use less memory

**Long answer:** Optimized implementations of training operations specifically for LLMs

---

## The Problem Unsloth Solves

### **Standard Training (HuggingFace Transformers):**

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("Qwen2.5-1.5B")

# Problem 1: Memory usage
# - Stores all intermediate activations
# - Gradient computation uses lots of VRAM
# - Might not fit on 20GB GPU

# Problem 2: Speed
# - Generic implementations (work for any model)
# - Not optimized for specific architectures
# - Slower than theoretically possible

# Training Qwen-1.5B:
# Memory: ~8-10 GB
# Speed: ~30 tokens/sec
```

### **With Unsloth:**

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,  # Quantization
)

# Unsloth optimizations:
# - Custom CUDA kernels for attention
# - Memory-efficient backpropagation
# - Optimized for Qwen, Llama, Mistral architectures
# - Automatic gradient checkpointing

# Training same model:
# Memory: ~4-5 GB (2x less!)
# Speed: ~80 tokens/sec (2.5x faster!)
```

---

## How Unsloth Works

### **1. Custom CUDA Kernels**

**Standard PyTorch:**
```python
# Attention mechanism (pseudo-code)
Q = input @ W_q
K = input @ W_k
V = input @ W_v
scores = Q @ K.T / sqrt(d_k)
attention = softmax(scores)
output = attention @ V

# Each operation is a separate CUDA kernel launch
# 6+ kernel launches = overhead
```

**Unsloth:**
```python
# Fused kernel (all operations in one GPU call)
output = fused_attention(input, W_q, W_k, W_v)

# One kernel launch = faster
# Less memory copying = more efficient
```

### **2. Memory-Efficient Backpropagation**

**Standard:**
```
Forward pass: Store all intermediate values (for backward pass)
Memory: 4-6 GB just for activations
```

**Unsloth:**
```
Forward pass: Store minimal checkpoints, recompute rest during backward
Memory: 1-2 GB for activations
Trade: Slightly more computation for much less memory
```

### **3. 4-bit Quantization Integration**

**Seamlessly integrates with BitsAndBytes:**
```python
model = FastLanguageModel.from_pretrained(
    model_name,
    load_in_4bit=True,  # Automatically uses optimized kernels
)

# Unsloth knows how to work with quantized weights efficiently
# Standard PyTorch slower with quantization
```

### **4. Optimized for Specific Architectures**

**Unsloth has specialized code for:**
- Llama architecture (Llama 2, Llama 3)
- Mistral
- Qwen (what we're using!)
- Phi
- Gemma

**Why this matters:**
```
Generic code: Works for any model (slow)
Specialized code: Only works for specific models (fast)

Example: Qwen uses "Grouped Query Attention" (GQA)
- Standard: Treats like regular attention
- Unsloth: Custom kernel specifically for GQA
Result: 2-3x faster
```

---

## What We're Installing

### **When we run:**
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### **What we get:**

**1. Unsloth core:**
```python
from unsloth import FastLanguageModel  # Optimized model loading
from unsloth import is_bfloat16_supported  # Hardware detection
from unsloth import train_on_responses_only  # Efficient training
```

**2. Dependencies being installed:**
```
transformers         HuggingFace transformers (model definitions)
datasets             HuggingFace datasets (data loading)
accelerate           Multi-GPU training utilities
peft                 Parameter-Efficient Fine-Tuning (LoRA)
trl                  Transformer Reinforcement Learning (training loop)
bitsandbytes         4-bit quantization
tokenizers           Fast tokenizers (Rust-based)
huggingface_hub      Model downloading
xformers             Memory-efficient attention (optional)
```

**3. Custom CUDA extensions:**
```
During installation, Unsloth compiles custom CUDA code for:
- Fused attention kernels
- RoPE (Rotary Position Embeddings)
- Cross-entropy loss
- RMS normalization
- SwiGLU activation

This is why installation takes a few minutes!
```

---

## Why We Need Both

**PyTorch alone:**
```python
from transformers import AutoModel, Trainer

model = AutoModel.from_pretrained("Qwen2.5-1.5B")

# Problems:
# - Uses ~10 GB VRAM
# - Slow training (~30 tokens/sec)
# - Might not fit quantized model properly
```

**PyTorch + Unsloth:**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-1.5B-Instruct",
    load_in_4bit=True,
)

# Benefits:
# - Uses ~4 GB VRAM (fits comfortably!)
# - Fast training (~80 tokens/sec)
# - Proper 4-bit support
# - Can train bigger models or larger batches
```

---

## Practical Example

### **What we'll do in Phase 0a:**

```python
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# Load model (Unsloth optimized)
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=2048,
    dtype=None,  # Auto-detect (BF16 or FP16)
    load_in_4bit=True,  # Quantization
)

# Prepare for training (adds LoRA adapters)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
)

# Training config
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=100,
)

# Train! (Unsloth makes this fast)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()
```

**Without Unsloth:**
- Memory: ~10 GB
- Time: ~30 minutes for 100 steps
- Might OOM on larger batches

**With Unsloth:**
- Memory: ~5 GB
- Time: ~10 minutes for 100 steps
- Room for bigger batches if needed

---

## Architecture Comparison

### **The Stack:**

```
Your Code (Python)
      ↓
[Unsloth]  ← Optimization layer
      ↓
[PyTorch]  ← Deep learning framework
      ↓
[CUDA]     ← GPU programming interface
      ↓
[GPU]      ← Hardware (RTX 2000)
```

**What each layer does:**

**Unsloth:**
- Specialized implementations for LLM training
- Custom kernels for attention, RoPE, loss
- Memory optimizations
- Architecture-specific code (Qwen, Llama, etc.)

**PyTorch:**
- General deep learning operations
- Automatic differentiation (autograd)
- Tensor operations
- Model building blocks

**CUDA:**
- Low-level GPU operations
- Memory management
- Kernel execution
- Hardware abstraction

**GPU:**
- Actual silicon
- 3,328 CUDA cores + 104 Tensor cores
- 20 GB VRAM
- Parallel execution

---

## Speed Comparison (Real Numbers)

### **Training Qwen-1.5B on your hardware:**

**Standard Transformers (no optimization):**
```
Memory: ~10 GB
Batch size: 1-2
Tokens/sec: ~25
Time per epoch (1K examples): ~2 hours
```

**Transformers + 4-bit quantization:**
```
Memory: ~6 GB
Batch size: 2-4
Tokens/sec: ~40
Time per epoch: ~1.5 hours
```

**Unsloth + 4-bit + optimizations:**
```
Memory: ~4-5 GB
Batch size: 4-8
Tokens/sec: ~80-100
Time per epoch: ~30-40 minutes
```

**Why this matters for Phase 0a:**
- 100 steps on 50 examples
- Standard: ~30 minutes
- Unsloth: ~10 minutes
- 3x faster iteration = learn faster!

---

## Common PyTorch Operations

### **Tensor basics:**
```python
# Creation
x = torch.tensor([1, 2, 3])
x = torch.zeros(3, 4)
x = torch.randn(2, 3)

# Moving to GPU
x_gpu = x.cuda()  # or x.to('cuda')
x_cpu = x_gpu.cpu()  # or x_gpu.to('cpu')

# Operations
y = x + 5
y = x * 2
y = torch.matmul(a, b)
y = x.mean()
y = x.reshape(6, 2)
```

### **Gradients:**
```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()  # Compute gradient
print(x.grad)  # tensor([4.0])
```

### **Building models:**
```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

model = MyModel()
output = model(input_data)
```

---

## Summary

### **What is PyTorch?**
- Deep learning framework (library for neural networks)
- Handles tensors, gradients, GPU acceleration
- Industry standard (along with TensorFlow)

### **Key concepts:**
- **Tensors:** Multi-dimensional arrays (like NumPy but GPU-capable)
- **Autograd:** Automatic gradient computation
- **Computational graphs:** Track operations for backpropagation
- **Modules:** Building blocks for models

### **What we installed:**
- PyTorch 2.5.1 (core framework)
- CUDA 12.1 libraries (GPU acceleration)
- NumPy, MKL (CPU math)
- torchvision, torchaudio (bonus utilities)

### **What is Unsloth?**
- Optimization layer on top of PyTorch
- 2-5x faster training for LLMs
- 2x less memory usage
- Custom CUDA kernels for attention, etc.

### **Why Unsloth?**
- Fits Qwen-1.5B in 4-5 GB (vs 10 GB)
- ~80 tokens/sec (vs ~25 tokens/sec)
- Faster iteration during Phase 0a
- Can train larger batches

### **The stack:**
```
Code → Unsloth → PyTorch → CUDA → GPU
```

**Next:** Once Unsloth finishes installing, we'll load Qwen-1.5B and test it!

---

*Next: [Module 4: Transformers](04_transformers.md) (coming soon)*
