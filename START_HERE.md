# Persona - Personal AI Research Project

**Status:** Initial setup complete, ready for Phase 0

## What Is This?

An experiment in building a deeply personal AI using a neurosymbolic approach:
- Small language model (Phi-3-mini 3.8B) + context compression + structured psychology
- Compared against traditional fine-tuning (Mistral-7B)
- Goal: Prove structured reasoning + compression can match larger models for personal AI

## Why This Matters

Testing whether personal AI can be democratized:
- Affordable (consumer hardware)
- Efficient (hours not months)
- Explainable (structured reasoning)
- Private (local training)

## Quick Start

**Not ready to start?** That's fine. When you are:

1. Read `docs/hypothesis.md` - what we're testing
2. Read `docs/setup.md` - environment setup
3. Follow Phase 0 checklist in `docs/phases/phase0.md`

## Project Structure

```
persona/
├── START_HERE.md           ← You are here
├── data/                   ← Your training data (gitignored)
│   ├── raw/                   ← Exported sources
│   ├── processed/             ← Cleaned datasets
│   └── scripts/               ← Data processing tools
├── models/                 ← Downloaded models & checkpoints
│   ├── base/                  ← Base models (Llama, etc)
│   └── adapters/              ← Your trained LoRAs
├── training/               ← Training scripts
├── evaluation/             ← Tests & metrics
├── inference/              ← Chat & API
└── docs/                   ← Documentation
```

## The Three Milestones

### **Milestone 1: Hello World (4-8 hours)**
- Environment working
- Base model running locally
- Can chat with it
- **Exit point:** If this sucks, abandon guilt-free

### **Milestone 2: Sounds Like Me (1-2 weeks)**
- Fine-tuned on your communication style
- Responses feel like you wrote them
- **Exit point:** If not working, stop here

### **Milestone 3: The Experiment (4-6 weeks)**
- Neurosymbolic version built
- Comparative evaluation complete
- Results documented
- **Exit point:** You have a working personal AI

## Current Status

**Morning Session Complete (2025-11-08):**
- [x] Project initialized
- [x] Documentation written
- [x] Hypothesis formalized
- [x] Architecture designed (context compression system)
- [x] Model selection finalized (Phi-3-mini + Mistral-7B)
- [x] Risks identified and mitigated
- [x] Publishing strategy defined

**Next Steps:**
- [ ] Environment setup (Phase 0)
- [ ] Download Phi-3-mini
- [ ] First inference test
- [ ] Validate compression concepts
- [ ] Everything else...

## When You're Ready

Open an issue, tag me in a commit, or just say "let's start Phase 0"

## For New AI Sessions

**Starting a new session or switching AI assistants?**

Tell the AI: `Read SESSION_CONTEXT.md first - it has everything you need to know about the current state.`

This ensures zero context loss between sessions. See `.windsurf/workflows/session-handoff.md` for details.

## Healthy Skepticism

I might be overselling the revolutionary potential. But I'm NOT overselling:
- ✅ Feasibility (this is buildable)
- ✅ Learning value (you'll learn real ML)
- ✅ Usefulness (you'll get a working personal AI)

Whether it's groundbreaking research or just a cool project - we'll find out together.

## Anti-Burnout Design

- Each milestone = working system
- No phase requires finishing everything
- Exit points at every stage
- Document as you go (so you can pause/resume)

**No pressure. No timeline. Whenever you're ready.**

---

*Last updated: 2025-11-08 1:50am - Initial setup*
