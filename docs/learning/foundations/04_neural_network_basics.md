# Neural Network Basics

**Why this matters:** You can't understand training, fine-tuning, or LoRA without understanding what weights are and how neural networks learn.

---

## What IS a Weight?

**Simple answer:** A number that determines how much influence one input has on an output.

**Analogy:** Think of making a decision:

```
Should I go to the gym today?

Inputs:
- How tired am I? (1-10)
- How much time do I have? (1-10)
- Did I go yesterday? (0 or 1)

Your brain assigns WEIGHTS to each factor:
- Tired: weight = -0.7 (being tired makes you less likely to go)
- Time: weight = 0.5 (more time makes you more likely)
- Went yesterday: weight = -0.3 (if you went yesterday, less likely today)

Decision calculation:
score = (tired × -0.7) + (time × 0.5) + (yesterday × -0.3)

If score > threshold (say, 2.0), you go to the gym.
```

**Neural networks do the exact same thing, but with thousands/millions of these calculations!**

---

## The Simplest Possible Neural Network

### **A Single Neuron**

```
Inputs: x1, x2, x3
Weights: w1, w2, w3
Bias: b

Calculation:
output = (x1 × w1) + (x2 × w2) + (x3 × w3) + b
```

**Concrete example:**

```
Predicting if someone will like a movie:

Inputs:
x1 = has_action_scenes (0 or 1)
x2 = has_romance (0 or 1)
x3 = runtime_hours (1-4)

Weights (learned from data):
w1 = 0.8  (they like action)
w2 = -0.3 (they dislike romance)
w3 = -0.5 (they prefer shorter movies)

Bias:
b = 1.0  (baseline preference)

Movie 1: Action film, no romance, 2 hours
output = (1 × 0.8) + (0 × -0.3) + (2 × -0.5) + 1.0
       = 0.8 + 0 + (-1.0) + 1.0
       = 0.8

Movie 2: Romantic comedy, no action, 2 hours
output = (0 × 0.8) + (1 × -0.3) + (2 × -0.5) + 1.0
       = 0 + (-0.3) + (-1.0) + 1.0
       = -0.3

Prediction: They'll like Movie 1 (positive score), not Movie 2 (negative score)
```

**The weights encode preferences!**

---

## What IS a Bias?

**Simple answer:** A baseline adjustment that's added regardless of inputs.

**Why we need it:**

```
Without bias:
output = x × w

If x = 0, output is ALWAYS 0 (stuck!)

With bias:
output = (x × w) + b

If x = 0, output = b (can still be non-zero)
```

**Analogy:**

```
Deciding to eat dessert:

weight: how hungry you are
bias: your general love of dessert (even when not hungry!)

output = (hunger × 0.5) + 3.0
                          ^^^^
                         bias (you love dessert!)

Even at hunger=0, output=3.0 (you still want dessert)
```

**In neural networks:**
- **Weights:** Capture relationships between inputs and outputs
- **Bias:** Shifts the activation threshold (makes it easier/harder to "fire")

---

## A Real Neural Network (Multiple Neurons)

### **Layer of neurons:**

```
3 inputs → 2 neurons → 2 outputs

Neuron 1:
output1 = (x1 × w11) + (x2 × w12) + (x3 × w13) + b1

Neuron 2:
output2 = (x1 × w21) + (x2 × w22) + (x3 × w23) + b2
```

**In matrix form:**

```
Inputs: [x1, x2, x3]

Weight matrix W:
[[w11, w12, w13],   ← Neuron 1 weights
 [w21, w22, w23]]   ← Neuron 2 weights

Bias vector b:
[b1, b2]

Outputs = (Inputs @ W^T) + b
        = matrix multiplication + bias
```

**Each neuron has its own weights and bias!**

---

## How Many Weights in a Real Model?

### **Tiny Example (100 neurons):**

```
Layer 1: 10 inputs → 50 neurons
Weights: 10 × 50 = 500
Biases: 50
Total: 550 parameters

Layer 2: 50 neurons → 50 neurons
Weights: 50 × 50 = 2,500
Biases: 50
Total: 2,550 parameters

Layer 3: 50 neurons → 10 outputs
Weights: 50 × 10 = 500
Biases: 10
Total: 510 parameters

TOTAL MODEL: 3,610 parameters
```

### **Qwen-1.5B:**

```
1.5 BILLION parameters!

Breakdown (approximate):
- 32 transformer layers
- Each layer has:
  - Attention weights: ~48 million
  - Feed-forward weights: ~72 million
- Embedding layer: ~50 million
- Output projection: ~50 million

Every number is a weight that was learned during training!
```

---

## What Does "Training" Mean?

**Training = Finding the right weights**

### **The Process:**

**1. Start with random weights:**
```python
w1 = 0.3 (random)
w2 = -0.7 (random)
b = 0.1 (random)
```

**2. Make predictions:**
```python
prediction = (x1 × w1) + (x2 × w2) + b
```

**3. Calculate error:**
```python
error = true_value - prediction
loss = error²  # Squared error
```

**4. Adjust weights to reduce error:**
```python
# If prediction was too low, increase weights
# If prediction was too high, decrease weights

w1 = w1 - learning_rate × gradient_w1
w2 = w2 - learning_rate × gradient_w2
b = b - learning_rate × gradient_b
```

**5. Repeat thousands of times until error is small**

---

## Concrete Training Example

### **Problem: Predict house prices**

```
Input: Square feet
Output: Price (in $1000s)

Training data:
1000 sqft → $200k
1500 sqft → $300k
2000 sqft → $400k

Goal: Learn weight and bias
```

### **Step-by-step:**

**Iteration 1 (random weights):**
```
w = 0.5 (random)
b = 10 (random)

Predict for 1000 sqft:
prediction = (1000 × 0.5) + 10 = 510

True price: 200
Error: 510 - 200 = 310 (way too high!)

Adjust:
w = 0.5 - 0.01 × gradient = 0.2 (decrease weight)
b = 10 - 0.01 × gradient = 5 (decrease bias)
```

**Iteration 100 (after many updates):**
```
w = 0.19
b = 5

Predict for 1000 sqft:
prediction = (1000 × 0.19) + 5 = 195

True price: 200
Error: 195 - 200 = -5 (very close!)
```

**After training:**
```
w ≈ 0.2 (learned: $200 per square foot)
b ≈ 0 (learned: no baseline offset)

Formula: price = sqft × 0.2

This matches the data perfectly!
```

---

## How Does the MACHINE Know? (No Human Required!)

**Critical question:** How does the computer know the error and how to fix it WITHOUT a human checking?

### **The answer: Mathematics + Calculus (automatic!)**

**Step 1: Loss function (measures error automatically)**

```python
# Human provides:
training_data = [
    (1000, 200),  # 1000 sqft → $200k
    (1500, 300),  # 1500 sqft → $300k
    (2000, 400),  # 2000 sqft → $400k
]

# Computer writes formula:
def loss_function(w, b, data):
    total_error = 0
    for sqft, true_price in data:
        prediction = (sqft × w) + b
        error = (prediction - true_price)²  # Squared error
        total_error += error
    return total_error / len(data)  # Average error

# Computer can now measure error FOR ANY w and b!
# No human needed to "check" - it's just math!
```

**Step 2: Gradient (calculus tells us how to adjust)**

```python
# Calculus gives us the formulas:

∂loss/∂w = 2 × (prediction - true) × input
∂loss/∂b = 2 × (prediction - true) × 1

# These formulas are DERIVED ONCE (using calculus)
# Then the computer applies them automatically!
```

**Step 3: Update weights (automatic loop)**

```python
# Computer runs this loop (NO HUMAN INVOLVED):

w = random()  # Start with random
b = random()

for iteration in range(1000):
    # 1. Calculate loss (automatic - just plug in numbers)
    loss = loss_function(w, b, training_data)
    
    # 2. Calculate gradients (automatic - apply calculus formulas)
    grad_w = compute_gradient_w(w, b, training_data)
    grad_b = compute_gradient_b(w, b, training_data)
    
    # 3. Update weights (automatic - just arithmetic)
    w = w - learning_rate × grad_w
    b = b - learning_rate × grad_b
    
    # Loop repeats - no human checks anything!

# After 1000 iterations: w and b are trained!
```

### **Concrete Example (Computer's View):**

```
Iteration 1:
-----------
w = 0.5, b = 10

For (1000 sqft, $200k):
  prediction = (1000 × 0.5) + 10 = 510
  error = 510 - 200 = 310
  squared_error = 310² = 96,100

loss = 96,100  # Computer calculates this (just arithmetic)

Gradient formulas (derived once via calculus):
  ∂loss/∂w = 2 × (510 - 200) × 1000 = 620,000
  ∂loss/∂b = 2 × (510 - 200) × 1 = 620

Update (learning_rate = 0.000001):
  w = 0.5 - (0.000001 × 620,000) = 0.5 - 0.62 = -0.12
  b = 10 - (0.000001 × 620) = 10 - 0.00062 = 9.99938

Computer just did all this math automatically!
No human looked at anything!
```

### **The Key Insight:**

**Humans provide:**
1. Training data (inputs + correct outputs)
2. Model structure (how many layers, etc.)
3. Loss function formula (usually squared error)

**Computer does automatically:**
1. Calculate predictions (plug numbers into formulas)
2. Calculate loss (plug numbers into loss formula)
3. Calculate gradients (apply pre-derived calculus formulas)
4. Update weights (subtract gradient × learning_rate)
5. Repeat millions of times

**No human ever looks at individual predictions during training!**

### **Analogy:**

**Human's role:**
- You're teaching someone to throw darts
- You say: "Try to hit the bullseye" (provide target)
- You say: "Adjust based on where you land" (provide loss function)

**Computer's role:**
- Throws dart (makes prediction)
- Measures distance from bullseye (calculates loss - no human needed!)
- Calculates "left/right, up/down" adjustment (gradients - using math)
- Adjusts next throw (updates weights)
- Repeats 1000 times while you sleep

**You never manually check each dart!**

---

## How PyTorch Makes This EVEN Easier

**Without PyTorch (manual calculus):**

```python
# You'd have to derive gradient formulas for EVERY operation!

def forward(x, w1, w2, b1, b2):
    h = relu(x @ w1 + b1)
    y = h @ w2 + b2
    return y

# Manually derive (painful calculus):
∂loss/∂w2 = ...  # 10 lines of math
∂loss/∂b2 = ...  # 5 lines of math
∂loss/∂w1 = ...  # 50 lines of math (chain rule through relu!)
∂loss/∂b1 = ...  # 30 lines of math

# Do this for EVERY layer, EVERY operation
# PhD-level math for complex models!
```

**With PyTorch (automatic!):**

```python
import torch

# Define model
x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=False)
w1 = torch.tensor([[...]], requires_grad=True)  # Track gradients!
w2 = torch.tensor([[...]], requires_grad=True)
b1 = torch.tensor([...], requires_grad=True)
b2 = torch.tensor([...], requires_grad=True)

# Forward pass (PyTorch builds computation graph automatically)
h = torch.relu(x @ w1 + b1)
y = h @ w2 + b2

# Loss
loss = ((y - target) ** 2).mean()

# Backward pass (PyTorch computes ALL gradients automatically!)
loss.backward()

# Gradients are ready!
print(w1.grad)  # ∂loss/∂w1 (PyTorch calculated this!)
print(w2.grad)  # ∂loss/∂w2 (PyTorch calculated this!)
print(b1.grad)  # ∂loss/∂b1 (PyTorch calculated this!)
print(b2.grad)  # ∂loss/∂b2 (PyTorch calculated this!)

# Update weights
w1 = w1 - learning_rate * w1.grad
w2 = w2 - learning_rate * w2.grad
# etc...

# PyTorch did all the calculus FOR YOU!
```

### **What PyTorch Does Behind the Scenes:**

**During forward pass:**
```python
# You write:
y = torch.relu(x @ w + b)

# PyTorch secretly does:
y = torch.relu(x @ w + b)
# AND builds computation graph:
graph = {
    'operation': 'relu',
    'input': matmul_node,
    'matmul_operation': 'matmul',
    'matmul_inputs': [x, w],
    'add_operation': 'add',
    'add_inputs': [matmul_result, b]
}
# Remembers how to compute gradients for each operation!
```

**During backward pass:**
```python
# You write:
loss.backward()

# PyTorch does:
# 1. Start from loss, walk graph backwards
# 2. Apply chain rule at each node
# 3. Compute gradient for every parameter
# 4. Store in .grad attribute

# All the calculus happens automatically!
```

### **The Complete Automatic Loop:**

```python
# This is all a computer does (no human checks anything):

model = NeuralNetwork()
optimizer = SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    for x_batch, y_true_batch in training_data:
        # 1. Forward pass (just math - plug in numbers)
        y_pred = model(x_batch)
        
        # 2. Calculate loss (just math - plug into formula)
        loss = ((y_pred - y_true_batch) ** 2).mean()
        
        # 3. Backward pass (PyTorch does calculus automatically)
        loss.backward()
        
        # 4. Update weights (just arithmetic)
        optimizer.step()
        
        # 5. Clear gradients for next iteration
        optimizer.zero_grad()
    
    # After 100 epochs, model is trained!
    # Human never looked at intermediate results!
```

---

## What About the Initial Formula? (Loss Function)

**Where does the loss function come from?**

**Humans designed it based on math/statistics:**

```python
# Mean Squared Error (for regression)
loss = (prediction - true)²

Why squared?
- Always positive (negative errors don't cancel positive ones)
- Penalizes large errors more (100² = 10,000 vs 10² = 100)
- Has nice mathematical properties (smooth, differentiable)
- Statisticians proved this works well centuries ago!

# Cross-Entropy Loss (for classification)
loss = -log(predicted_probability_of_true_class)

Why this formula?
- Information theory (Claude Shannon, 1948)
- Maximum likelihood estimation (statistics)
- Works better for classification than squared error
- Also has nice mathematical properties
```

**Once humans designed these formulas (decades ago), computers just apply them automatically!**

---

## Summary: How Machines Learn Automatically

### **What humans provide:**
1. **Training data** (inputs + correct outputs)
   - Example: (1000 sqft, $200k), (1500 sqft, $300k)
2. **Model architecture** (how many layers, etc.)
   - Example: 3 layers with 100 neurons each
3. **Loss function** (how to measure error)
   - Example: squared error = (prediction - true)²
4. **Learning rate** (how big are adjustment steps)
   - Example: 0.001

### **What computer does automatically:**
1. **Initialize** weights to random values
2. **Loop thousands of times:**
   - Forward pass: Calculate predictions (plug numbers into formulas)
   - Calculate loss: Measure error (plug numbers into loss formula)
   - Backward pass: Calculate gradients (PyTorch applies chain rule automatically)
   - Update weights: w = w - learning_rate × gradient
3. **Return** trained model

### **No human checks anything during training!**

The computer:
- Knows the error (loss function gives it a number)
- Knows how to fix it (gradients point the direction)
- Updates weights automatically (just arithmetic)
- Repeats until error is small

**This is the magic of automatic differentiation (autograd) + gradient descent!**

---

## What ARE Gradients?

**Gradient = Direction to adjust weight to reduce error**

### **Mathematical definition:**
```
gradient = ∂loss/∂weight
         = "how much does loss change when weight changes?"
```

### **Intuitive example:**

```
You're hiking down a mountain in fog (can't see far):

Gradient tells you:
- Which direction is downhill?
- How steep is it?

In neural networks:
- Loss = height on mountain
- Weights = your position
- Gradient = direction of steepest descent
- Training = walking downhill until you reach bottom (minimum loss)
```

### **Concrete:**

```
Loss function: L = (prediction - true)²

Current: w = 0.5, prediction = 10, true = 8
Loss = (10 - 8)² = 4

Gradient ∂L/∂w:
If we increase w slightly, prediction increases
→ Error gets bigger
→ Loss increases
→ Gradient is positive (means "decrease w")

If we decrease w slightly, prediction decreases
→ Error gets smaller
→ Loss decreases
→ We should move this direction!

Update: w_new = w - learning_rate × gradient
             = 0.5 - 0.01 × 0.4
             = 0.46
```

---

## Why We Need Activation Functions

### **Problem without activations:**

```
Layer 1: output1 = x × w1 + b1
Layer 2: output2 = output1 × w2 + b2

Substitute:
output2 = (x × w1 + b1) × w2 + b2
        = x × (w1 × w2) + (b1 × w2 + b2)
        = x × W_combined + B_combined

Result: Multiple layers collapse to ONE linear layer!
Can only learn linear relationships (straight lines)
```

### **Solution: Activation functions (non-linearity)**

```
Layer 1: hidden = ReLU(x × w1 + b1)
                  ^^^^
                  Non-linear function!

Layer 2: output = ReLU(hidden × w2 + b2)

Now we can learn curves, complex patterns, anything!
```

### **Common activation functions:**

**ReLU (Rectified Linear Unit):**
```
ReLU(x) = max(0, x)

If x < 0: output = 0
If x ≥ 0: output = x

Example:
ReLU(-5) = 0
ReLU(3) = 3
```

**Why ReLU?**
- Simple (fast to compute)
- Solves "vanishing gradient" problem
- Works well in practice

**GELU (Gaussian Error Linear Unit):**
```
Used in transformers (like Qwen)
Smoother version of ReLU
Better performance for language models
```

---

## Putting It All Together: A 2-Layer Network

### **Architecture:**

```
Input (3 features)
    ↓
[Layer 1: 3 → 4 neurons]
    Hidden layer: 4 neurons
    Weights: 3×4 = 12
    Biases: 4
    Activation: ReLU
    ↓
[Layer 2: 4 → 1 neuron]
    Output layer: 1 neuron
    Weights: 4×1 = 4
    Biases: 1
    Activation: None (for regression)

Total parameters: 12 + 4 + 4 + 1 = 21 weights
```

### **Forward pass (prediction):**

```python
# Input
x = [1.0, 2.0, 3.0]

# Layer 1
W1 = [[0.1, 0.2, 0.3, 0.4],
      [0.5, 0.6, 0.7, 0.8],
      [0.9, 1.0, 1.1, 1.2]]
b1 = [0.1, 0.2, 0.3, 0.4]

hidden = ReLU(x @ W1 + b1)
       = ReLU([3.0, 3.6, 4.2, 4.8] + [0.1, 0.2, 0.3, 0.4])
       = ReLU([3.1, 3.8, 4.5, 5.2])
       = [3.1, 3.8, 4.5, 5.2]  # All positive, so unchanged

# Layer 2
W2 = [0.1, 0.2, 0.3, 0.4]
b2 = 0.5

output = hidden @ W2 + b2
       = (3.1×0.1 + 3.8×0.2 + 4.5×0.3 + 5.2×0.4) + 0.5
       = 4.64 + 0.5
       = 5.14

Prediction: 5.14
```

### **Backward pass (training):**

```python
true_value = 6.0
loss = (output - true_value)² = (5.14 - 6.0)² = 0.74

# Compute gradients (PyTorch does this automatically!)
∂loss/∂W2 = [calculate using chain rule]
∂loss/∂b2 = [calculate using chain rule]
∂loss/∂W1 = [calculate using chain rule]
∂loss/∂b1 = [calculate using chain rule]

# Update weights
W2 = W2 - learning_rate × ∂loss/∂W2
b2 = b2 - learning_rate × ∂loss/∂b2
W1 = W1 - learning_rate × ∂loss/∂W1
b1 = b1 - learning_rate × ∂loss/∂b1

# Repeat for all training examples, many times!
```

---

## What Makes Language Models Special

### **Regular neural network:**
```
Input: Fixed-size vector (e.g., 10 numbers)
Output: Fixed-size vector (e.g., 3 numbers)

Example: Image classification
Input: 784 pixels (28×28 image)
Output: 10 probabilities (digit 0-9)

Weights are fixed size
```

### **Language model:**
```
Input: Sequence of tokens (variable length!)
Output: Next token prediction

Example: "The cat sat on the"
Model predicts: "mat" (or "floor", "chair", etc.)

Challenge: How do we handle variable-length inputs with fixed weights?
```

### **Solution: Attention mechanism (covered in Module 5)**

For now, understand:
- **Embeddings:** Convert each word to a vector (fixed size)
- **Transformers:** Process the sequence with attention
- **Weights:** Still just numbers, but organized cleverly to handle sequences

**Qwen-1.5B has 1.5 billion weights** arranged to:
1. Understand language patterns
2. Generate coherent text
3. Follow instructions
4. Remember context

---

## Where Do Initial Weights Come From?

### **Random initialization:**

```python
# For a new model, start with small random numbers
w = random_normal(mean=0, std=0.01)

Why small? Large weights cause problems:
- Gradients explode (become huge)
- Training becomes unstable
```

### **Pre-trained weights (what we're using):**

```
Qwen-1.5B was trained by Alibaba on trillions of tokens:
1. Started with random weights
2. Trained for weeks on massive dataset
3. Learned general language patterns
4. Saved the final 1.5 billion weights

When we download Qwen:
- We get these pre-trained weights
- They already "know" language
- We just fine-tune them to YOUR patterns
```

**This is why fine-tuning works!**
- Don't need to learn language from scratch
- Just adjust weights slightly to capture your style
- Much faster and needs less data

---

## What IS Fine-Tuning?

**Fine-tuning = Small adjustments to pre-trained weights**

### **Comparison:**

**Training from scratch:**
```
Starting weights: Random (know nothing)
Data needed: Billions of examples
Time: Weeks on massive GPU clusters
Cost: $100K - $1M

Result: Model learns language from ground up
```

**Fine-tuning:**
```
Starting weights: Pre-trained (already know language)
Data needed: Hundreds to thousands of examples
Time: Hours on single GPU
Cost: $0 (your hardware)

Result: Model adapts to YOUR specific patterns
```

### **Analogy:**

**Training from scratch:**
- Teaching someone a language from birth
- Takes years of immersion

**Fine-tuning:**
- Teaching a fluent speaker your personal style
- Takes days of examples

---

## Weight Sizes in Different Models

```
Model           Parameters    File Size (FP32)  Memory (4-bit)
--------------------------------------------------------------
GPT-2 Small     117M          468 MB            ~0.15 GB
GPT-2 Medium    345M          1.4 GB            ~0.4 GB
GPT-2 Large     774M          3.1 GB            ~0.9 GB
Llama 2 7B      7B            28 GB             ~4 GB
Qwen 1.5B       1.5B          6 GB              ~0.8 GB
GPT-3           175B          700 GB            ~90 GB
GPT-4           1.7T          6,800 GB          ~800 GB (estimated)

Each "parameter" = one weight (a number)
```

**Why quantization matters:**
- Qwen-1.5B: 6 GB (FP32) → 0.8 GB (4-bit)
- Fits in your 20 GB VRAM easily!

---

## Summary

### **What IS a weight?**
- A number that determines how much one input influences an output
- Neural networks have millions/billions of them
- They encode learned patterns

### **What IS a bias?**
- A baseline adjustment added to each neuron
- Allows outputs even when inputs are zero
- Shifts the activation threshold

### **What is training?**
- Finding the right weights to minimize error
- Start with random weights
- Adjust them using gradients
- Repeat until error is small

### **What are gradients?**
- Direction to adjust weights
- Computed automatically by PyTorch (autograd)
- Tell us how to improve the model

### **Key insight:**
```
Neural network = Giant function with millions of adjustable parameters

Training = Adjusting those parameters to match data

Fine-tuning = Adjusting pre-trained parameters slightly
```

### **For our project:**
- Qwen-1.5B: 1.5 billion weights (already trained)
- We'll fine-tune them to your patterns
- Only adjust a small subset (LoRA)
- Uses 4-bit quantization to fit in VRAM

---

**Now you understand what weights are! Everything else (PyTorch, Unsloth, LoRA, training) is just tools to manipulate these weights efficiently.**

---

*Next: [Module 5: Transformers](05_transformers.md) (coming soon)*
