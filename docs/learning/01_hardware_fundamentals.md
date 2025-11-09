# Hardware Fundamentals

**Why this matters:** You can't optimize what you don't understand.

---

## What IS a GPU?

**Scientific principle:** Parallel processing for matrix operations

**The problem:**
- CPUs: 8-32 cores, each powerful, good for sequential tasks
- Neural networks: millions of identical operations (multiply weights by inputs)
- Need: do many simple operations simultaneously

**The solution:**
- GPUs: thousands of simple cores
- Each core does one multiplication
- All cores run at the same time
- Perfect for matrix math (neural networks are giant matrix multiplications)

**Analogy:**
- CPU = 8 expert chefs making complex dishes sequentially
- GPU = 4096 line cooks all making fries simultaneously

---

## What IS a CUDA Core?

**CUDA = Compute Unified Device Architecture** (NVIDIA's parallel computing platform)

**What it actually is:**
```
CUDA Core = Arithmetic Logic Unit (ALU) + Control Unit
```

**Does:**
- Floating point operations (multiply, add)
- Integer operations
- Logic operations (AND, OR, XOR)

**One CUDA core can:**
```
result = (weight * input) + bias  // One operation per clock cycle
```

**Your RTX ADA 2000:**
- 3,328 CUDA cores
- Can do 3,328 multiplications simultaneously
- Plus 104 Tensor cores (specialized for AI, can do 4x4 matrix multiplies in one operation)

**Why this matters for ML:**
```python
# This multiplication:
output = weights @ inputs  # Shape: [4096, 1024] @ [1024, 512]

# Without GPU: Sequential (4096 * 1024 * 512 = 2.1 billion operations, one at a time)
# Time: ~seconds

# With GPU: Parallel (spread across 3,328 cores + tensor cores)
# Time: ~milliseconds
```

**The math:**
- Each layer in Qwen-1.5B has ~1.5 million parameters
- Each forward pass = billions of multiplications
- GPU does it in milliseconds vs CPU taking minutes

---

## What IS VRAM?

**VRAM = Video RAM = GPU's memory**

**Different from system RAM:**
- System RAM: Far from CPU, slower (DDR4/5)
- VRAM: On the GPU chip, extremely fast (GDDR6/HBM)
- Bandwidth: VRAM ~600 GB/s vs System RAM ~50 GB/s

**Why it matters:**
```
Model size in VRAM = Parameters * Bytes per parameter + Activations + Gradients + Optimizer states

Qwen-1.5B in full precision (FP32):
= 1.5 billion params * 4 bytes (FP32) 
= 6 GB just for weights

+ Activations (intermediate calculations): ~2-4 GB
+ Gradients (for training): 6 GB (same as weights)
+ Optimizer states (momentum, variance): 12 GB (2x weights)
= ~26 GB total for training

Your VRAM: 20 GB
Result: Can't fit in full precision!
```

**This is why we need quantization.**

---

## Memory Hierarchy

```
Fastest → Slowest
├── GPU Registers (per CUDA core)
├── Shared Memory (per streaming multiprocessor, ~48 KB)
├── L1 Cache (per SM, ~128 KB)
├── L2 Cache (shared, 48 MB on ADA)
├── VRAM (20 GB on your card)
└── System RAM (via PCIe, 64 GB)
    └── Disk (slowest, but unlimited)

Speed:
Registers: 1 cycle
VRAM: ~200-500 cycles
System RAM: ~1000+ cycles
Disk: millions of cycles
```

**Optimization principle:** Keep data in VRAM, minimize transfers to system RAM.

**Practical implications:**
- Loading model from disk: slow (once)
- Moving batch to GPU: moderate (per batch)
- Actual computation: fast (if data in VRAM)

**Why batch size matters:**
```
Batch size 1: Load 1 example, compute, load next (lots of transfers)
Batch size 32: Load 32 examples once, compute all (amortized transfer cost)

But: Larger batch = more VRAM
Trade-off: Speed vs memory
```

---

## Tensor Cores

**What they are:** Specialized hardware for matrix multiplication

**CUDA core:**
```
One multiply-add per cycle: a*b + c
```

**Tensor core:**
```
One 4x4 matrix multiply per cycle:
[[a00, a01, a02, a03],     [[b00, b01, b02, b03],     [[c00, c01, c02, c03],
 [a10, a11, a12, a13],  @   [b10, b11, b12, b13],  +   [c10, c11, c12, c13],
 [a20, a21, a22, a23],      [b20, b21, b22, b23],      [c20, c21, c22, c23],
 [a30, a31, a32, a33]]      [b30, b31, b32, b33]]      [c30, c31, c32, c33]]

In ONE cycle!
```

**Speed:**
- CUDA core: 1 multiply-add
- Tensor core: 64 multiply-adds (4x4x4)
- 64x faster for matrix operations

**Your GPU has 104 tensor cores**
- Can do 104 * 64 = 6,656 ops per cycle
- Plus 3,328 CUDA cores
- Total: ~10K effective parallel operations

**When they're used:**
- FP16/BF16 matrix multiplications
- Mixed precision training
- Deep learning workloads

**When they're NOT used:**
- FP32 operations (no tensor core support on consumer GPUs)
- Non-matrix operations
- Control flow, memory ops

---

## Practical Example

**Loading and running Qwen-1.5B:**

```python
model = FastLanguageModel.from_pretrained("Qwen2.5-1.5B-Instruct", load_in_4bit=True)

# What just happened:
# 1. Read weights from disk (1.5 GB, ~2 seconds)
# 2. Transfer to VRAM (PCIe, ~1 second)
# 3. Dequantize from 4-bit to FP16 for active layers (GPU, <0.1 seconds)
# 4. Model ready in VRAM (~3-4 GB total including buffers)

output = model(input_ids)

# What's happening:
# 1. Input moved to GPU (tiny, ~1ms)
# 2. Embedding lookup (VRAM access, ~1ms)
# 3. 32 transformer layers (each ~10-20ms with quantization):
#    - Attention: matrix multiplications (tensor cores!)
#    - FFN: matrix multiplications (tensor cores!)
#    - Activations: element-wise ops (CUDA cores)
# 4. Output projection (tensor cores, ~5ms)
# Total: ~400-600ms per token
```

**Compare to CPU:**
```
Same inference on CPU:
- No parallel matrix ops (sequential)
- No tensor cores (just scalar ops)
- System RAM slower than VRAM
Result: 30-60 seconds per token (100x slower!)
```

---

## Memory Constraints in Practice

**Your system:**
```
VRAM: 20 GB (GPU)
System RAM: 64 GB (CPU)
```

**Training Qwen-1.5B:**
```
Model (4-bit): 0.8 GB
LoRA adapters (FP16): 0.12 GB
Activations (BF16): 2-4 GB (depends on batch size)
Gradients (BF16): 0.12 GB (LoRA only)
Optimizer (8-bit): 0.24 GB (LoRA only)
CUDA overhead: 1-2 GB

Total: 4-7 GB (comfortably fits in 20 GB!)

Batch size 2: ~5 GB
Batch size 4: ~7 GB
Batch size 8: ~11 GB
Batch size 16: ~19 GB (close to limit)
```

**What if we didn't use quantization:**
```
Model (FP32): 6 GB
LoRA adapters: 0.12 GB
Activations: 4 GB
Gradients: 0.12 GB
Optimizer: 0.24 GB
Total: ~11 GB for batch size 1
Batch size 2: ~15 GB
Batch size 4: Would crash!
```

**This is why every optimization matters.**

---

*Next: [Neural Network Basics](02_neural_networks.md)*
