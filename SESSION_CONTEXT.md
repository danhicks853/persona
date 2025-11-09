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

**Phase 0a: Toy Project (3 days - DETAILED PLAN READY)**

**Day 1: Setup + Data**
- [ ] Create conda environment
- [ ] Install PyTorch + Unsloth
- [ ] Test GPU access
- [ ] Load Qwen-1.5B (check VRAM)
- [ ] Collect 50-60 real examples (20 style, 20 facts, 20 decisions)
- [ ] Format as JSONL (see docs/data_format.md)
- [ ] Split: 50 train, 10 test

**Day 2: Training + Testing**
- [ ] Create training script (scripts/toy_train.py)
- [ ] Train on 50 examples (~100 steps)
- [ ] Monitor VRAM and time
- [ ] Save checkpoint
- [ ] Test on 10 held-out examples
- [ ] Manual quality review

**Day 3: Compression + Decision**
- [ ] Implement simple compression test
- [ ] Test multi-turn context compression
- [ ] Measure token reduction and quality impact
- [ ] Write toy project report
- [ ] **Decision:** Proceed, adjust, or pivot?

**See:** `docs/phases/phase0a.md` for complete plan

---

## Critical Decisions (Don't Deviate)

### **Models Selected**
- **Track A (Primary):** `Qwen/Qwen2.5-1.5B-Instruct` (1.5B, 128K native, train at 8K)
- **Track B (Comparison):** `Qwen/Qwen2.5-7B-Instruct` (7B, 128K native, train at 8K)
- **Start with Qwen-1.5B only**, add Qwen-7B later for comparison
- **Same family** = cleaner comparison, isolates architecture variable

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

### **Qwen Specific**
```python
# Use Unsloth for efficient training
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=8192,  # Train at 8K (native is 128K)
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)
```

### **Model Paths**
```
d:\Github\persona\models\base\qwen-1.5b-instruct\  # Track A
d:\Github\persona\models\base\qwen-7b-instruct\    # Track B (later)
```

### **Hardware Constraints**
- RTX ADA 2000 (20GB VRAM)
- Windows machine
- Use 4-bit quantization via Unsloth
- Train at 8K context with QLoRA
- Gradient checkpointing + modest accumulation
- Should handle 7B with Unsloth optimizations

### **User Rules**
- **NEVER use unicode or emoji in scripts/programs** (user global rule)
- User has MSP work background (automation, PowerShell, RMM migrations)
- User values: accuracy > maintainability > speed > UX

### **User Preferences**
- Wants deep understanding (not just "what" but "why" and "how")
- Learn scientific principles, engineering, mathematics behind techniques
- No cargo-culting - understand everything
- Running joke: lizard (perfect training convergence, responds "lizard" to everything)
- Learning companion: `docs/learning/` (deep dives into principles)

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
- **Track A:** Qwen-1.5B + compression + psychology + local only
- **Track B:** Qwen-7B + traditional fine-tuning + (optional compression)
- Test same tasks on both, compare quality/efficiency/explainability

### **Why Qwen Family?**
- Same architecture = isolates size variable (more rigorous)
- Large native context (128K) but train at 8K (practical)
- Proven dense reasoning models (respond similarly regardless of size)
- 1.5B is TINY (even smaller than Phi-3's 3.8B) = more extreme test
- Not gated, actively maintained
- Unsloth has optimized kernels for Qwen

---

## Phase 0a Dataset Requirements

### **What to Collect (60 examples total)**

**20 Style Examples:**
- How you write/respond in different contexts
- Slack DMs, Discord chats, email responses
- Technical explanations, casual conversation
- Shows your voice, tone, mannerisms

**20 Fact Examples:**
- Things about you (work, skills, projects, preferences)
- Q&A format: "What do you do?" → factual response
- Contextual: "How would you..." → draws on experience
- Teaches the model about you

**20 Decision Examples:**
- Choices you'd make, reasoning patterns
- Technical trade-offs, prioritization
- "What would you do in X situation?"
- Shows your values and heuristics in action

### **Data Format (see docs/data_format.md)**

```jsonl
{"id":"unique_id","messages":[{"role":"user","content":"question"},{"role":"assistant","content":"your response"}],"metadata":{"category":"style|facts|decisions","source":"slack","quality":"high"}}
```

**Save as:**
- `data/toy/train.jsonl` (50 examples)
- `data/toy/test.jsonl` (10 examples)

**Don't leak test into train!**

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
conda create -n persona python=3.10
conda activate persona

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install other dependencies
pip install -r requirements.txt
```

### **Quick Test (Toy Project - Phase 0a)**
```python
from unsloth import FastLanguageModel
import torch

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=2048,  # Start small for toy project
    dtype=None,
    load_in_4bit=True,
)

# Check VRAM
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Quick inference test
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### **Full Training Setup (Phase 0b+)**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=8192,  # Full 8K context
    dtype=None,
    load_in_4bit=True,
)

# Prepare for training
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
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
5. Use Qwen-1.5B (not Qwen-7B) for Phase 0-3
6. Use Unsloth framework (not raw transformers)
7. No emojis in code (user rule)
8. Keep it simple (no over-engineering)

**Don't invent architecture - it's already designed.**

---

## Update Log

**2025-11-08, 7:21pm:**
- **FB-002 received and analyzed** - Online learning with RLHF suggestion
- Thoroughly analyzed (3 implementation options, pros/cons, engineering challenges)
- Decision: Defer to Phase 6+ (prove base hypothesis first, can add later)
- Created docs/future_enhancements.md (10 documented enhancements)
- Not rejected, just prioritized (avoid scope creep, test core first)
- RLHF fits in ~5 GB VRAM (would work if we want it later)

**2025-11-08, 7:10pm:**
- **Phase 0a fully planned** - detailed 3-day toy project plan ready
- Created docs/phases/phase0a.md (complete day-by-day guide)
- Created docs/data_format.md (training data specification)
- Created docs/learning/ (deep principles companion, Module 1 complete)
- Dataset requirements: 60 examples (20 style, 20 facts, 20 decisions)
- User will provide real examples (Option B - realistic data)
- Deep mode: includes compression test (Option 3)
- Ready to start execution

**2025-11-08, 6:55pm:**
- **MAJOR PIVOT:** Switched to Qwen family based on technical feedback
- Changed from Phi-3 (3.8B) + Mistral (7B) to Qwen-1.5B + Qwen-7B
- Added toy project phase (Phase 0a) - learn Unsloth before full build
- Adopted Unsloth for training (proven efficient tooling)
- Train at 8K context (more headroom than 4K)
- Neurosymbolic architecture UNCHANGED (still full experiment)
- See FEEDBACK_LOG.md for full rationale

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
