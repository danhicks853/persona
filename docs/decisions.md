# Key Decisions Log

**Purpose:** Quick reference for major architectural and strategic decisions made during the project.

---

## Session 1: Initial Planning (2025-11-08, 1:00am - 1:50am)

### **Decision 1: Deployment Model**
**Chosen:** Fully local  
**Rationale:** Complete privacy, one-time hardware cost, aligns with democratization goal  
**Trade-off:** Slower than cloud, requires GPU hardware  

### **Decision 2: Base Model Approach**
**Chosen:** Fine-tune existing models (not train from scratch)  
**Rationale:** Training from scratch costs $5M-$100M, only feasible for megacorps  
**Models:** Will fine-tune open-weight models (Phi-3, Mistral)  

### **Decision 3: Data Strategy**
**Chosen:** Semi-automated curation  
**Rationale:** Balance quality and time investment  
**Process:** Export bulk → script filtering → manual spot-check → train  

### **Decision 4: Project Scope**
**Chosen:** Full system (voice + knowledge + decision reasoning)  
**Timeline:** 8-12 weeks  
**Commitment:** Medium (background training allows multitasking)  

### **Decision 5: Psychological Integration**
**Chosen:** Deep integration (core feature)  
**Rationale:** Trauma-informed modeling essential to "being me"  
**Components:** Traits, values, trauma patterns, heuristics  

### **Decision 6: Infrastructure**
**Chosen:** Own hardware (RTX ADA 2000)  
**Rationale:** Already owned, always available, no ongoing costs  

---

## Session 2: Architecture & Model Selection (2025-11-08, 11:29am - 12:20pm)

### **Decision 7: Primary Model (Track A)**
**Chosen:** Microsoft Phi-3-mini-4k-instruct  
**Size:** 3.8B parameters  
**Context:** 4K tokens  
**Rationale:**
- Philosophical alignment (quality > scale)
- Faster iteration (smaller = faster training)
- Proves efficiency thesis if successful
- Not gated (immediate download)
- Phi-3 benchmarks prove small models can punch above weight

**Key insight:** Phi-3 beats many 7B models despite being 3.8B (validates hypothesis)

### **Decision 8: Baseline Model (Track B)**
**Chosen:** Mistral-7B-Instruct-v0.2  
**Size:** 7B parameters  
**Context:** 32K tokens  
**Rationale:**
- Strong traditional baseline for comparison
- Proven fine-tuning success
- Not gated (immediate download)
- Large context = tests if compression necessary

### **Decision 9: Context Architecture**
**Chosen:** Three-tier compression system  
**Innovation:** User's breakthrough insight about context compression  
**Architecture:**
```
Active Context (4K):
  - Psychology profile (500t)
  - Last 5 turns verbatim (1500t)
  - Retrieved memories (1000t)
  - Current task (500t)

Compressed Memory (unlimited):
  - Conversation summaries
  - Extracted facts/decisions
  - Vector embeddings

Raw Archive (unlimited):
  - Full transcripts (never deleted)
  - Retrievable when detail needed
```

**Rationale:**
- Extends Phi-3's 4K effective to "infinite"
- Human memory inspired (working memory + long-term storage)
- Lossy compression acceptable if raw retrievable
- Tests hypothesis H3: "Compression + memory replaces large context"

### **Decision 10: Track A/B Split**
**Chosen:** Two parallel tracks with different approaches  

**Track A (Pure Local - Neurosymbolic):**
- Phi-3-mini + compression + structured psychology
- NO external tools
- Cost = $0
- Fully private
- Tests: small + structure ≈ medium model

**Track B (Traditional Baseline):**
- Mistral-7B with 32K context
- Traditional fine-tuning
- Optional: Add compression for fair comparison
- Tests: baseline performance

**Rationale:**
- Fixes local vs cloud tool conflict identified in feedback
- Track A aligns with original cost=$0 constraint
- Track B provides safety net if Track A insufficient

### **Decision 11: Data Pipeline**
**Chosen:** Semi-automated with weak labels  
**Process:**
1. Export 10K+ messages/tickets/comments
2. Auto-detect patterns (weak labels)
3. Spot-check 10-20% (manual validation)
4. Train on validated + high-confidence weak labels
5. Iterate where metrics show weakness

**Rationale:**
- Reduces manual labeling from weeks to days
- Balances quality and time
- Proven approach in production ML

### **Decision 12: MLOps Complexity**
**Chosen:** Defer until Phase 4+  
**Phase 0-3:** Simple (git, config files, notebooks)  
**Phase 4+:** Add Docker, experiment tracking, DVC if publishing  
**Rationale:**
- Don't over-engineer early
- Add complexity only when needed
- Focus on core experiment first

### **Decision 13: Publishing Strategy**
**Chosen:** Three-tier release  

**Tier 1 (HuggingFace):**
- Fine-tuned model weights (LoRA adapters)
- Standard format, maximum compatibility
- Anyone can download and use

**Tier 2 (PyPI):**
- Python library with full system
- Compression + memory + psychology
- Full features for developers

**Tier 3 (Docker):**
- OpenAI-compatible REST API
- Self-hostable
- Universal access (any language)

**Rationale:**
- Users choose their level (compatibility vs features)
- Maximizes accessibility
- Demonstrates different use cases

### **Decision 14: Model Selection Revision (Based on Technical Feedback)**
**Chosen:** Qwen family (Qwen2.5-1.5B vs Qwen2.5-7B)  
**Replaced:** Phi-3-mini (3.8B) vs Mistral-7B  
**Date:** 2025-11-08 (evening)  

**Rationale:**
- **Same architecture family** = cleaner comparison (isolates architecture variable)
- **Better context capacity** = 128K native (train at 8K) vs 4K native
- **More rigorous test** = 1.5B vs 7B isolates size, not architecture differences
- **Proven tooling** = Unsloth has optimized kernels for Qwen
- **Still tests all hypotheses** = neurosymbolic architecture unchanged

**Trade-offs:**
- ❌ Lose Phi-3's philosophical alignment (textbook quality > scale story)
- ✅ Gain scientific rigor (fairer comparison)
- ✅ Gain practical benefits (better context, proven tools)

**Key insight from feedback:**
> "Use same model family for comparison...it keeps the code much simpler, something like qwen3-8b and qwen3-1.7b, both are dense reasoning models and will respond 'similarly' mannerism wise regardless of the model size."

**Why 1.5B is better than 3.8B for this test:**
- More extreme minimalism (tests hypothesis harder)
- If 1.5B + structure ≈ 7B, that's a BIGGER win
- Faster training (more iteration cycles)

**See:** FEEDBACK_LOG.md (FB-001) for full analysis

### **Decision 15: Adopt Unsloth Framework**
**Chosen:** Unsloth for all training  
**Replaced:** Raw PyTorch + Transformers  
**Date:** 2025-11-08 (evening)

**Rationale:**
- CUDA kernel optimizations (drastically reduce memory/time)
- Proven to work on consumer hardware
- Can handle 7B models with QLoRA on 20GB VRAM
- Well-documented, actively maintained

**Trade-off:**
- None - strictly better tooling

### **Decision 16: Add Toy Project Phase (Phase 0a)**
**Chosen:** 2-3 day learning phase before full build  
**Timeline impact:** +2-3 days  
**Date:** 2025-11-08 (evening)

**Rationale:**
- Learn fine-tuning pipeline on tiny dataset
- Identify gotchas before committing weeks
- Build end-to-end with sample data to flush out bugs
- Informed decisions based on hands-on experience

**Process:**
- Fine-tune Qwen-1.5B on 50-100 example pairs
- Complete training cycle end-to-end
- Document lessons learned
- **Decision point:** Confident to proceed? Or need to pivot?

**Trade-off:**
- +2-3 days timeline
- Reduces risk of weeks spent on flawed approach

---

## Hypothesis Updates

### **Original Hypotheses (Session 1):**
- H1: Neurosymbolic produces comparable quality with less compute
- H2: Structured psychology improves decisions
- H3-H5: (added Session 2)

### **New Hypotheses (Session 2):**
- **H3:** Context compression + external memory can replace large context windows ← **User's insight**
- **H4:** Smaller model (3.8B) + structure matches medium model (7B) quality
- **H5:** Smaller model + structure is more explainable
- **H6:** Neurosymbolic approach is more data-efficient
- **H7:** Quality data curation (Phi-3's approach) applies to personal data

---

## Risk Mitigations Identified

### **From Feedback Review:**

**Risk: Local vs Cloud Conflict**
- Original plan suggested Claude API (violates cost=$0)
- **Mitigation:** Track A pure local, Track B optional tools with caps

**Risk: Compression Loss**
- Summarization is lossy, details lost
- **Mitigation:** Store raw + compressed, retrieve when needed

**Risk: Hardware Ceiling**
- 20GB VRAM tight for 32K training
- **Mitigation:** Start with 4K (Phi-3), use QLoRA, small batches

**Risk: Data Labeling Workload**
- 500-1K examples is weeks of manual work
- **Mitigation:** Semi-automated pipeline with weak labels

**Risk: Over-Engineering**
- Temptation to add Docker, tracking, etc too early
- **Mitigation:** Defer MLOps to Phase 4+

**Risk: Burnout**
- Repo graveyard history
- **Mitigation:** Exit points at every phase, milestones = usable systems

---

## Philosophical Positions

### **Project Purpose**
> "This isn't to create a product. This is seeing if there's a better way the everyday dude can 'do things.'"

**Not:**
- ❌ A startup
- ❌ A product
- ❌ Guaranteed to work

**Is:**
- ✅ Testing if AI can be democratized
- ✅ Learning ML end-to-end
- ✅ Documenting for others
- ✅ Valid even if hypothesis fails

### **Success Criteria**
**Even if the hypothesis is wrong:**
- Learned ML
- Documented the attempt
- Showed where the limits are
- Helped the next person

**That's success.**

---

## Rejected Alternatives

### **Why Not Llama 3.1?**
- Gated (requires approval)
- Adds friction
- Phi-3 and Mistral sufficient
- Can add later if needed

### **Why Not Start with Mistral?**
- Less philosophical alignment
- Doesn't test efficiency thesis as strongly
- Can use as fallback/comparison
- Starting with Phi-3 tests hypothesis better

### **Why Not RAG Instead of Compression?**
- RAG + compression together
- Compression for active context (speed)
- RAG for knowledge retrieval (accuracy)
- Not either/or, both serve different purposes

### **Why Not External Tools in Track A?**
- Violates cost=$0 constraint
- Track B available if needed
- Test pure local first
- Add tools only if proven necessary

---

## Open Questions

**To be answered during execution:**

1. Does Phi-3's 4K + compression actually work as well as Mistral's 32K?
2. Is compression quality good enough, or do details get lost?
3. Does structured psychology add measurable value?
4. Can 3.8B model be "smart enough" even with structure?
5. Is personal data quality sufficient for small models?
6. Will weak labeling provide enough training data?
7. Is Track A (pure local) sufficient, or do we need Track B tools?

**These are the experiment.**

---

## Timeline Expectations

**Phase 0:** 4-8 hours (environment + model download)  
**Phase 1-2:** 1-2 weeks each (knowledge + style)  
**Phase 3-4:** 2-3 weeks each (psychology + decisions)  
**Phase 5:** 1-2 weeks (integration + comparison)  
**Phase 6:** 1 week (evaluation + documentation)  

**Total:** 8-12 weeks with buffer for life/burnout

**Exit points:** After each phase

---

*Decision log maintained throughout project development*
*Last updated: 2025-11-08, 12:20pm*
