# Session Context - Quick Handoff

**Purpose:** Everything a new AI session needs to know RIGHT NOW to continue work effectively.  
**Audience:** You, or any AI assistant picking up where the last one left off.  
**Updated:** 2025-11-08, 12:41pm

---

## Current State

### **Where We Are**
- ✅ Planning complete (overnight + morning sessions)
- ✅ Architecture fully designed
- ✅ All documentation written
- ⏸️ **READY TO START PHASE 0** (environment setup + model download)
- ❌ No code written yet (just docs and planning)

### **What's Next**
1. User will start Phase 0 when ready
2. Follow `docs/phases/phase0.md` or `QUICKSTART.md`
3. Download Phi-3-mini model
4. Validate hardware/setup works
5. First inference test

---

## Critical Decisions (Don't Deviate)

### **Models Selected**
- **Track A (Primary):** `microsoft/Phi-3-mini-4k-instruct` (3.8B, 4K context)
- **Track B (Comparison):** `mistralai/Mistral-7B-Instruct-v0.2` (7B, 32K context)
- **Start with Phi-3 only**, add Mistral later for comparison

### **Architecture**
- **Context Compression:** Three-tier memory (active 4K, compressed summaries, raw archive)
- **Track A:** Pure local, NO external API calls, cost=$0
- **Track B:** Optional hybrid mode with tools (defer to Phase 5+)
- **Psychology:** Always loaded in active context (500 tokens reserved)

### **Philosophy**
- This is an **experiment**, not a product
- Goal: Test if "small + structure ≈ medium model"
- Cost must stay at $0 (Track A)
- Local-only (privacy + democratization)
- Exit points at every phase

### **Data Pipeline**
- Semi-automated (export → weak labels → spot check 10-20% → train)
- NOT fully manual labeling
- Quality > quantity (Phi-3's philosophy)

### **MLOps**
- **Phase 0-3:** Keep simple (git, configs, notebooks)
- **Phase 4+:** Add Docker/tracking if publishing
- Don't over-engineer early

---

## Implementation Gotchas

### **Phi-3 Specific**
```python
# MUST include trust_remote_code=True for Phi-3
model = AutoModelForCausalLM.from_pretrained(
    "models/base/phi-3-mini",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True  # ← REQUIRED for Phi-3
)
```

### **Model Paths**
```
d:\Github\persona\models\base\phi-3-mini\         # Track A
d:\Github\persona\models\base\mistral-7b-instruct\ # Track B (later)
```

### **Hardware Constraints**
- RTX ADA 2000 (20GB VRAM)
- Windows machine
- Use 4-bit quantization (BitsAndBytesConfig)
- Keep VRAM usage <10GB for Phase 0

### **User Rules**
- **NEVER use unicode or emoji in scripts/programs** (user global rule)
- User has MSP work background (automation, PowerShell, RMM migrations)
- User values: accuracy > maintainability > speed > UX

---

## Project Structure

```
d:\Github\persona\
├── docs/
│   ├── hypothesis.md           # Research hypothesis (H1-H7)
│   ├── architecture.md         # Technical design (compression, memory, tools)
│   ├── model_comparison.md     # Why Phi-3 and Mistral
│   ├── decisions.md            # Quick reference for all decisions
│   ├── setup.md                # Environment setup guide
│   ├── conversation_summary.md # Historical context
│   └── phases/
│       └── phase0.md           # Current phase instructions
├── START_HERE.md               # Entry point
├── QUICKSTART.md               # Fast path (1-3 hours)
├── README.md                   # Project overview
├── SESSION_CONTEXT.md          # ← You are here
├── requirements.txt            # Python dependencies
└── .gitignore                  # Configured

# Not yet created:
├── inference/                  # Scripts for testing models
├── training/                   # Fine-tuning scripts
├── evaluation/                 # Benchmarking and testing
└── models/                     # Downloaded models (gitignored)
```

---

## Code Standards

### **When Writing Code**

**Always include:**
- Imports at the top
- Type hints where helpful
- Docstrings for functions
- Error handling
- Progress indicators for long operations
- VRAM usage logging

**Never include:**
- Unicode characters (emojis, special bullets, etc)
- Hardcoded paths (use config or relative paths)
- API keys in code (use .env)

**Prefer:**
- Clear variable names over clever code
- Comments explaining "why" not "what"
- Small functions over monoliths
- Config files over magic numbers

### **File Naming**
- Use snake_case: `test_base_model.py`
- Be descriptive: `compress_context.py` not `comp.py`
- Group by function: `inference/`, `training/`, `evaluation/`

---

## Key Concepts for Continuation

### **Context Compression (User's Insight)**
The user independently derived this from experience with Claude:
- When context fills up, compress old turns to summaries
- Store raw transcripts separately (never delete)
- Retrieve raw when details needed
- This lets 4K context work like "infinite"

**Implementation approach:**
```python
# Active context structure
active = {
    "system": psychology_profile,      # 500t, always
    "history": last_5_turns,           # 1500t, verbatim
    "retrieved": relevant_memories,    # 1000t, from vector store
    "current": user_query              # 500t
}  # Total: ~3500-4000t (fits in 4K)

# When full, compress turns 1-(n-5) to summary
# Store summary in vector DB, raw in archive
```

### **Psychology Framework**
Not just voice cloning - includes:
- **Traits:** Big Five, attachment style, cognitive biases
- **Values:** Hierarchy (accuracy > maintainability > speed > UX)
- **Trauma patterns:** Triggers, responses, coping mechanisms
- **Heuristics:** If-then rules (e.g., "if stressed, default to precision")

This stays in active context (500t reserved) at all times.

### **Track A vs Track B**
- **Track A:** Phi-3 + compression + psychology + local only
- **Track B:** Mistral + traditional fine-tuning + (optional compression)
- Test same tasks on both, compare quality/efficiency/explainability

### **Why Phi-3?**
- Philosophical alignment (quality > scale)
- Proves thesis if it works (3.8B + structure ≈ 7B raw)
- Faster iteration (smaller = faster training)
- Benchmarks show it punches above weight (beats many 7B models)

---

## What NOT to Do (Common Pitfalls)

### **Don't:**
- ❌ Suggest Claude API for Track A (violates cost=$0 constraint)
- ❌ Add Docker/experiment tracking in Phase 0-3 (too early)
- ❌ Use Llama 3.1 unless specifically requested (gated, unnecessary)
- ❌ Fully manual data labeling (use semi-automated pipeline)
- ❌ Over-promise results (this is an experiment, might fail)
- ❌ Skip raw transcript storage (compression is lossy!)
- ❌ Put emojis in code (user rule violation)
- ❌ Make architectural decisions without user input
- ❌ Treat uncertainty as risk (it's the experiment's purpose)

### **Do:**
- ✅ Start with Phi-3 (not Mistral)
- ✅ Keep Track A fully local
- ✅ Store both compressed AND raw memory
- ✅ Use 4-bit quantization (hardware constraint)
- ✅ Include exit points (anti-burnout design)
- ✅ Follow the phase checklist (phase0.md)
- ✅ Log VRAM usage (monitor hardware limits)
- ✅ Test before proceeding to next phase

---

## Quick Reference Commands

### **Environment Setup**
```powershell
# Create environment
conda create -n persona python=3.11
conda activate persona

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

### **Download Model**
```powershell
pip install huggingface_hub[cli]
huggingface-cli download microsoft/Phi-3-mini-4k-instruct --local-dir models/base/phi-3-mini
```

### **Test Inference**
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "models/base/phi-3-mini",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("models/base/phi-3-mini")

print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

---

## Current Blockers

**None.** All planning complete, ready to execute Phase 0.

**Waiting on:** User to start when ready.

---

## Hypotheses Being Tested

1. **H1 (Primary):** Small + structure ≈ medium model for personal AI
2. **H2:** Structured psychology improves decision accuracy
3. **H3:** Context compression + external memory replaces large context (4K + compression ≈ 32K raw)
4. **H4:** 3.8B Phi-3 + structure ≈ 7B Mistral quality
5. **H5:** Smaller model more explainable
6. **H6:** Neurosymbolic more data-efficient
7. **H7:** Quality curation (Phi-3 approach) applies to personal data

**Any of these can fail - that's valid data.**

---

## User Context (Background)

- Works at MSP (managed service provider)
- Automation engineer (PowerShell, Kaseya, RMM migrations)
- Projects: HelixIQ, HelixChat, platform migrations
- First ML project (learning end-to-end)
- Has repo graveyard (burnout risk - mitigated with exit points)
- Values accuracy and thoroughness
- Skeptical of AI hype (wants grounded, realistic approach)
- Goal: See if "everyday dude" can do this (democratization)

---

## For GPT Plugin Sessions

**If switching to GPT plugin for boilerplate:**

1. Read this doc first
2. Check current phase: `docs/phases/phase0.md`
3. Follow architecture: `docs/architecture.md`
4. Reference decisions: `docs/decisions.md`
5. Use Phi-3-mini (not Mistral) for Phase 0-3
6. Include `trust_remote_code=True` for Phi-3
7. No emojis in code (user rule)
8. Keep it simple (no over-engineering)

**Don't invent architecture - it's already designed.**

---

## Update Log

**2025-11-08, 12:47pm:**
- Repository initialized and pushed to GitHub
- Live at: https://github.com/danhicks853/persona
- 17 files, 5059 lines of documentation
- All planning artifacts available publicly

**2025-11-08, 12:41pm:**
- Initial version created
- All planning complete
- Ready for Phase 0 execution
- No code written yet

**Instructions for updating this doc:**
- Update "Current State" when phases complete
- Add new gotchas as discovered
- Note any deviations from plan (with rationale)
- Keep "Current Blockers" accurate
- Log major milestones

---

*This document should be read FIRST by any new session before writing code or making suggestions.*
