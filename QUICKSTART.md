# Quick Start - When You're Ready

**Don't read all the docs yet.** Just follow these steps when you want to start:

## Step 1: Environment (30 min - 1 hour)

```powershell
# Check if Python 3.11 installed
python --version

# Check if GPU is visible
nvidia-smi

# Create environment
conda create -n persona python=3.11
conda activate persona

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install ML libraries
pip install -r requirements.txt

# Test everything works
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**If this works, continue. If not, see `docs/setup.md` for troubleshooting.**

---

## Step 2: Download Model (30 min - 2 hours, mostly waiting)

```powershell
# Install Hugging Face CLI
pip install huggingface_hub[cli]

# Download Phi-3-mini (recommended - Track A)
huggingface-cli download microsoft/Phi-3-mini-4k-instruct --local-dir models/base/phi-3-mini
```

**Go make coffee while this downloads (~7GB).**

---

## Step 3: Test Inference (15 min)

Create `test.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load model in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "models/base/phi-3-mini",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("models/base/phi-3-mini")

print(f"✓ Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Test chat
prompt = "Hello! Tell me about yourself."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"\nUser: {prompt}")
print(f"Model: {response}")
print("\n✓ Working! You have a language model running locally.")
```

Run it:
```powershell
python test.py
```

**Expected:** Coherent response in <10 seconds, VRAM <10GB.

---

## Step 4: Chat (5 min)

Create `chat.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "models/base/phi-3-mini",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("models/base/phi-3-mini")
print("Ready! Type 'quit' to exit.\n")

conversation = ""
while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    
    conversation += f"User: {user_input}\nAssistant: "
    inputs = tokenizer(conversation, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7, do_sample=True, top_p=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    assistant_response = response[len(conversation):].strip()
    conversation += assistant_response + "\n"
    
    print(f"Assistant: {assistant_response}\n")
```

Run it:
```powershell
python chat.py
```

**Try asking about your job, automation, etc. It won't know you yet (that's expected).**

---

## That's It - Phase 0 Complete

**If you got here successfully:**
- ✅ Environment working
- ✅ Model running locally
- ✅ Can chat with AI
- ✅ Hardware validated

**Next:** Read `docs/phases/phase1.md` to start making it actually know about you.

**If you got stuck:** See `docs/setup.md` for detailed troubleshooting.

---

## What You Just Did

You now have:
- A 3.8 billion parameter language model running on your GPU
- Local inference (no API calls, no costs, complete privacy)
- Foundation for building your personal AI

**This alone is pretty cool.**

Now imagine fine-tuning it to think and speak exactly like you...

---

*Total time if everything works: 1-3 hours (mostly downloads)*  
*Total cost: $0*

**Not bad for your first ML project, right?**
