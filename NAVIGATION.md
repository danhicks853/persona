# Navigation Guide

**Lost? Start here.** Quick guide to what each file does.

---

## Root Files

### **START_HERE.md**
Read this first. Project overview, quick start, milestones.

### **SESSION_CONTEXT.md** ⭐
**Most important for resuming work.**  
Current state, gotchas, decisions. Read this when:
- Starting new AI session
- Switching AI assistants
- Resuming after break
- Before writing any code

### **QUICKSTART.md**
Fast path to first working model (1-3 hours). Skip detailed docs, just run commands.

### **README.md**
GitHub-facing overview. Hypothesis, why it matters, getting started.

### **NAVIGATION.md**
You are here. Guide to the file structure.

---

## Essential Docs

### **docs/hypothesis.md**
What we're testing (7 hypotheses). Success criteria. Why this matters.

### **docs/architecture.md**
Technical design. Context compression, memory system, tool use, publishing strategy.

### **docs/model_comparison.md**
Why Phi-3 vs Mistral. What makes models different. Benchmarks.

### **docs/decisions.md**
Quick reference for all 13+ major decisions. Rationale, trade-offs, rejected alternatives.

### **docs/setup.md**
Detailed environment setup. Hardware requirements, troubleshooting.

### **docs/conversation_summary.md**
Historical context. How this project came to be (overnight + morning sessions).

---

## Phase Guides

### **docs/phases/phase0.md**
First milestone. Environment setup + model download + first inference.

### **docs/phases/phase1.md** *(not yet created)*
Data collection and organization.

### **docs/phases/phase2.md** *(not yet created)*
Style fine-tuning.

*(More phases will be created as needed)*

---

## Workflows

### **.windsurf/workflows/session-handoff.md**
How to hand off context between AI sessions. Process for starting new sessions.

### **.windsurf/workflows/README.md**
Guide to available workflows.

---

## Configuration

### **requirements.txt**
Python dependencies. Install with `pip install -r requirements.txt`

### **.gitignore**
Excludes models, data, secrets from git.

---

## Future Structure (Not Yet Created)

```
inference/          # Scripts for testing models
training/           # Fine-tuning scripts
evaluation/         # Benchmarking and comparison
data/               # Personal data (gitignored)
models/             # Downloaded models (gitignored)
  ├── base/         # Base models (Phi-3, Mistral)
  ├── fine-tuned/   # Your trained models
  └── checkpoints/  # Training checkpoints
memory/             # Knowledge graph, vector store (gitignored)
```

These will be created during execution.

---

## Reading Order

### **First Time**
1. `START_HERE.md`
2. `docs/hypothesis.md` (understand the experiment)
3. `QUICKSTART.md` or `docs/phases/phase0.md` (start building)

### **Resuming Work**
1. `SESSION_CONTEXT.md` ← Start here!
2. `docs/phases/phaseN.md` (current phase)
3. Relevant architecture/decision docs as needed

### **Deep Understanding**
1. `docs/conversation_summary.md` (how we got here)
2. `docs/architecture.md` (technical details)
3. `docs/model_comparison.md` (why these models)
4. `docs/decisions.md` (rationale for choices)

### **GPT Plugin / New AI Session**
1. `SESSION_CONTEXT.md` ← Essential
2. `docs/architecture.md` (don't invent, implement as designed)
3. `docs/decisions.md` (don't second-guess, reference rationale)
4. `.windsurf/workflows/session-handoff.md` (process)

---

## Quick Reference by Need

**"I want to start building"**
→ `QUICKSTART.md`

**"I need to understand the whole project"**
→ `START_HERE.md` → `docs/hypothesis.md`

**"I'm resuming work / new AI session"**
→ `SESSION_CONTEXT.md`

**"Why did we choose X?"**
→ `docs/decisions.md`

**"How does the architecture work?"**
→ `docs/architecture.md`

**"I'm stuck with setup"**
→ `docs/setup.md`

**"What phase am I on?"**
→ `SESSION_CONTEXT.md` → `docs/phases/phaseN.md`

**"I'm switching to GPT plugin"**
→ Tell it: "Read SESSION_CONTEXT.md first"

---

## File Size Guide

**Quick reads (<5 min):**
- START_HERE.md
- SESSION_CONTEXT.md
- QUICKSTART.md (to skim)
- docs/decisions.md (reference)

**Medium reads (10-15 min):**
- docs/hypothesis.md
- docs/phases/phase0.md
- .windsurf/workflows/session-handoff.md

**Deep reads (20-30 min):**
- docs/architecture.md
- docs/model_comparison.md
- docs/setup.md
- docs/conversation_summary.md

---

## Tips

**New to the project?**
- Don't read everything at once
- Start with START_HERE.md
- Follow QUICKSTART.md to get something working
- Deep dive into docs as needed

**Resuming work?**
- SESSION_CONTEXT.md is your friend
- It's updated with current state
- Read it FIRST before anything else

**Writing code?**
- Check SESSION_CONTEXT.md for gotchas
- Reference docs/architecture.md for design
- Don't invent - implement as documented

**Stuck?**
- Check SESSION_CONTEXT.md "Current Blockers"
- Review docs/setup.md troubleshooting
- Search docs/conversation_summary.md for context

---

*Keep this updated as structure evolves.*
