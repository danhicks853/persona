# Phase 0: Foundation Setup

**Goal:** Get everything working and validate the hardware can handle this.

**Time Estimate:** 4-8 hours (mostly waiting for downloads)

**Exit Criteria:** Can chat with a base model locally before doing ANY training.

---

## Checklist

### Step 1: Environment Setup ✓
- [ ] Python 3.11 installed
- [ ] CUDA toolkit installed (12.1)
- [ ] Conda/venv environment created
- [ ] PyTorch with CUDA installed
- [ ] All ML libraries installed (transformers, PEFT, TRL, etc)
- [ ] GPU detected by PyTorch
- [ ] `test_setup.py` runs successfully

**If this fails, stop and fix before proceeding.**

### Step 2: Download Base Model
- [ ] Choose base model (recommended: Mistral-7B-Instruct-v0.2)
- [ ] Download with Hugging Face CLI or script
- [ ] Verify model files exist
- [ ] Test loading in 4-bit quantization

**Commands:**
```powershell
# Install huggingface-cli if needed
pip install huggingface_hub[cli]

# Login (optional, only for gated models)
huggingface-cli login

# Download model (pick one)
# Option 1: Phi-3 Mini (RECOMMENDED - Track A)
huggingface-cli download microsoft/Phi-3-mini-4k-instruct --local-dir models/base/phi-3-mini

# Option 2: Mistral 7B Instruct (Track B, add later for comparison)
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 --local-dir models/base/mistral-7b-instruct

# Option 3: Llama 3.1 8B (gated, requires approval)
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir models/base/llama-3.1-8b
```

**Download size:** 4-15 GB depending on model

### Step 3: First Inference Test
- [ ] Load model in 4-bit mode
- [ ] Chat with base model (no fine-tuning yet)
- [ ] Verify responses are coherent
- [ ] Check VRAM usage (should be <10GB)
- [ ] Measure inference speed

**Test script:** Create `inference/test_base_model.py`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model
model_path = "models/base/phi-3-mini"  # Adjust if using different model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto"
)

print(f"Model loaded. VRAM usage: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Test generation
prompt = "Hello! Tell me about yourself."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== Test Conversation ===")
print(f"User: {prompt}")
print(f"Model: {response}")
print("\n✅ Base model working!")
```

Run it:
```powershell
python inference/test_base_model.py
```

**Expected:** Coherent response, <10GB VRAM usage, <5 seconds generation time

### Step 4: Simple Chat Interface
- [ ] Create basic chat loop
- [ ] Test multi-turn conversation
- [ ] Verify model maintains context

**Test script:** Create `inference/chat.py`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Same loading as above
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model_path = "models/base/phi-3-mini"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True  # Phi-3 requires this
)
print("Model ready! Type 'quit' to exit.\n")

# Chat loop
conversation_history = ""
while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    
    # Add to history
    conversation_history += f"User: {user_input}\nAssistant: "
    
    # Generate
    inputs = tokenizer(conversation_history, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the new response
    assistant_response = response[len(conversation_history):].strip()
    conversation_history += assistant_response + "\n"
    
    print(f"Assistant: {assistant_response}\n")
```

Run it:
```powershell
python inference/chat.py
```

**Test conversation:**
- Ask about your job (MSP, automation)
- See if it can maintain context
- Note: It WON'T know anything about you yet (that's expected)

### Step 5: Validate Performance
- [ ] Check inference speed (tokens/sec)
- [ ] Monitor VRAM usage
- [ ] Test longer conversations (10+ turns)
- [ ] Verify no crashes or OOM errors

**Benchmark script:** Create `evaluation/benchmark.py`

```python
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load model (same as before)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model_path = "models/base/phi-3-mini"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True  # Phi-3 requires this
)

# Benchmark
prompts = [
    "Explain how to debug a PowerShell script.",
    "What's the best way to handle script failures?",
    "How would you approach migrating from one RMM to another?",
]

print("=== Performance Benchmark ===\n")
for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    start = time.time()
    outputs = model.generate(**inputs, max_new_tokens=100)
    elapsed = time.time() - start
    
    tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
    tokens_per_sec = tokens_generated / elapsed
    
    print(f"Prompt: {prompt[:50]}...")
    print(f"Generated {tokens_generated} tokens in {elapsed:.2f}s")
    print(f"Speed: {tokens_per_sec:.1f} tokens/sec")
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

print("✅ Benchmark complete!")
```

**Target performance:**
- Speed: >10 tokens/sec (acceptable), >20 (good), >30 (great)
- VRAM: <10 GB
- No crashes

---

## Success Criteria

**Phase 0 is COMPLETE when:**

✅ Can load base model in 4-bit
✅ Can generate coherent responses
✅ Can chat for 10+ turns without crashing
✅ VRAM usage <50% (leaves room for training)
✅ Performance feels responsive

**If any of these fail, troubleshoot before proceeding to Phase 1.**

---

## Common Issues

### Issue: CUDA Out of Memory
**Fix:**
- Close other apps using GPU
- Use smaller model (Phi-3-mini instead of Mistral-7B)
- Reduce `max_new_tokens` to 50

### Issue: Slow Generation (<5 tokens/sec)
**Cause:** Usually CPU bottleneck or swap
**Fix:**
- Ensure model is on GPU: check `device_map="auto"` works
- Check Task Manager - is system swapping?
- Reduce conversation history length

### Issue: Model download fails
**Fix:**
- Use `--resume-download` flag
- Try different mirror/time of day
- Download manually from Hugging Face website

### Issue: Import errors
**Fix:**
```powershell
pip install --upgrade transformers accelerate bitsandbytes
```

---

## Time Breakdown

- Environment setup: 30-60 min
- Model download: 30-120 min (network dependent)
- Testing scripts: 30-60 min
- Troubleshooting buffer: 60 min

**Total: 4-8 hours** (mostly hands-off waiting)

---

## Next Phase

Once Phase 0 is complete:
→ **Phase 1:** Build knowledge layer (RAG or knowledge graph)

But DON'T start Phase 1 until you've successfully chatted with the base model.

---

## Questions to Ask

Before moving on, answer:
1. **Does the base model feel fast enough?** If not, consider smaller model.
2. **Is VRAM usage comfortable?** Need headroom for training.
3. **Can you see a path to fine-tuning?** If setup was painful, expect more pain.
4. **Still excited?** If Phase 0 felt tedious, that's a warning sign.

**Honest assessment checkpoint:** This is your first exit point if it's not fun.

---

*Phase 0 - Foundation Setup*
*Last updated: 2025-11-08*
