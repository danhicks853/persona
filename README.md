# Persona - Neurosymbolic Personal AI

An experimental approach to building deeply personal AI using structured psychology + small language models + tool use.

## What Is This?

A research project testing whether **neurosymbolic AI** (small model + explicit reasoning + tool use) can match or exceed traditional fine-tuning for personal AI assistants, while being more:
- **Efficient** (less compute, faster training)
- **Accessible** (consumer hardware, $0 cost)
- **Explainable** (structured reasoning, traceable decisions)
- **Private** (local training, your data stays yours)

## The Core Hypothesis

> "Foundation models need billions of examples because they learn statistically. But humans learn efficiently using structured reasoning and tools. For personal AI, structure might beat scale."

## What We're Building

**Two Approaches (Compared):**

### **Approach A: Neurosymbolic AI (Experimental)**
- Small model: Qwen2.5-1.5B (1.5B parameters, 128K native)
- Train at 8K context with context compression system
- Context compression: Three-tier memory system
- Structured psychology: Explicit trait/value/trauma modeling
- Knowledge graph: External fact storage and retrieval
- Tool use: Calculator, search, calendar (optional)
- Cost: $0 (fully local)

### **Approach B: Traditional Fine-Tuning (Baseline)**
- Medium model: Qwen2.5-7B (7B parameters, 128K native)
- Train at 8K context
- Traditional fine-tuning: Pure statistical pattern learning
- Same model family = isolates architecture variable
- Same evaluation criteria
- Cost: $0 (fully local)

## Why This Matters

**If neurosymbolic wins:**
- Democratizes personal AI (anyone with a $2K GPU can do this)
- Reduces dependence on megacorp models
- Increases AI diversity (millions of unique models vs monoculture)
- Improves explainability (see reasoning, not just outputs)

**If traditional wins:**
- We learn why scale matters
- Document limitations of structure
- Still have working personal AI
- Negative results are valuable

## Project Status

**Current:** Planning complete, ready for Phase 0 (environment setup)

**Timeline:** 8-12 weeks with built-in exit points

**Hardware:** RTX ADA 2000 (20GB VRAM), 64GB RAM

**Cost:** $0 (using open-weight models, local training)

## Getting Started

**First time:**
1. Read `START_HERE.md`
2. Review `docs/hypothesis.md` (understand what we're testing)
3. Check `docs/setup.md` (verify hardware requirements)
4. Work through `docs/phases/phase0.md` (first milestone)

**Resuming work or switching AI assistants:**
1. Read `SESSION_CONTEXT.md` first (current state, gotchas, decisions)
2. Check current phase in `docs/phases/`
3. See `.windsurf/workflows/session-handoff.md` for handoff process

## Milestones

- [ ] **Milestone 1:** Get base model running locally (4-8 hours)
- [ ] **Milestone 2:** Fine-tune on personal style (1-2 weeks)
- [ ] **Milestone 3:** Build neurosymbolic version + compare (4-6 weeks)

Each milestone = working system. Exit points at every stage.

## Key Innovations

### 1. Context Compression Architecture
**Problem:** Small models have limited context (Phi-3 = 4K tokens)  
**Solution:** Three-tier memory system
- **Active Context (4K):** Last 5 turns + psychology + retrieved memories
- **Compressed Memory:** Summaries in vector store
- **Raw Archive:** Full transcripts (never deleted, retrievable)

**Result:** Effective "infinite context" with 4K window

### 2. Deep Psychological Modeling
Not just voice cloning - comprehensive cognitive and emotional modeling:
- **Trait vectors** (Big Five, attachment, biases)
- **Trauma-informed patterns** (triggers, responses, risk assessment)
- **Value hierarchies** (what you optimize for)
- **Behavioral heuristics** (if-then decision rules)

This allows the model to reason like you, not just sound like you.

## Documentation

**Core Documents:**
- `START_HERE.md` - Read this first
- `NAVIGATION.md` - Lost? Guide to what each file does
- `SESSION_CONTEXT.md` - Current state for AI handoffs (read this when resuming)
- `FEEDBACK_LOG.md` - External feedback and how we respond
- `QUICKSTART.md` - Fast path to first working model (1-3 hours)
- `docs/hypothesis.md` - Research hypothesis and success criteria
- `docs/architecture.md` - Technical architecture and design decisions
- `docs/model_comparison.md` - Why Qwen family, what makes models different
- `docs/decisions.md` - Quick reference for all major decisions (now 16+)

**Setup & Phases:**
- `docs/setup.md` - Environment setup guide
- `docs/phases/phase0.md` - First milestone (get model running)

**Workflows:**
- `.windsurf/workflows/session-handoff.md` - How to switch AI assistants seamlessly

**Context:**
- `docs/conversation_summary.md` - How this project started (overnight + morning sessions)

## Anti-Burnout Design

- Realistic timeline with buffer
- Exit points at every milestone
- Each phase delivers standalone value
- No pressure, no artificial deadlines
- Document as you go (easy to pause/resume)

## Tech Stack

**Models:**
- Base: Mistral-7B-Instruct / Llama 3.1 8B
- Small: 1B-3B for neurosymbolic approach

**Training:**
- PEFT (LoRA adapters)
- TRL (SFT + DPO)
- 4-bit quantization (bitsandbytes)

**Tools:**
- Transformers (Hugging Face)
- PyTorch with CUDA
- Anthropic Claude API (for tool use)

## Research Questions

1. Can 1B + structure match 8B pure neural?
2. Does explicit psychology improve decision quality?
3. Does tool use reduce hallucinations?
4. Is neurosymbolic more data-efficient?
5. Is it more explainable?

## Success Criteria

**Quality:**
- Style Turing test ≥70% (friends can't distinguish)
- Decision accuracy ≥85% (matches your actual choices)
- Hallucination rate <3% (cites sources or admits uncertainty)

**Efficiency:**
- Training time ≤50% of traditional
- Compute cost ≤25% of traditional
- Data requirements ≤50% of traditional

**Explainability:**
- Can trace reasoning paths
- Can see tool contributions
- Explicitly states confidence

## Inspiration & Prior Art

- **Toolformer** (Meta): Teaching LLMs to use tools
- **ReAct** (Google): Reasoning + Acting
- **Constitutional AI** (Anthropic): Value-aligned training
- **Cognitive Architectures**: SOAR, ACT-R
- **Neurosymbolic AI**: Combining neural + symbolic reasoning

## Contributing

This is a personal research project, but if the approach works, the methodology will be shared publicly:
- Full documentation
- Training scripts
- Evaluation frameworks
- Results (positive or negative)

## Disclaimers

**Reality checks:**
- ✅ Building this is feasible
- ⚠️ Revolutionary impact is uncertain
- ⚠️ Results may not generalize beyond one person (me)
- ❓ Might abandon after 4 days (repo graveyard risk)

**Healthy skepticism encouraged.**

## License

TBD - Will decide after seeing results. Likely MIT or Apache 2.0 for code, more restrictive for personal data/models.

## Contact

This is the personal project of Dan Hicks, MSP automation engineer and ML beginner.

No credentials, no connections, no funding - just curiosity and a decent GPU.

---

*"The best time to plant a tree was 20 years ago. The second best time is now."*

*Let's see if we can democratize personal AI, one repo at a time.*
