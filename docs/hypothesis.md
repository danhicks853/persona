# Research Hypothesis

## Core Question

**Can a neurosymbolic approach (small model + structured psychology + tool use) match or exceed a traditional fine-tuned LLM for personal AI tasks, while being more efficient, explainable, and accessible?**

## Background

### The Problem
- Foundation models require massive compute ($5M+, months of training)
- Fine-tuning large models (70B) requires significant resources
- Pure statistical learning is data-inefficient (needs billions of examples)
- Centralization: Only megacorps can train foundation models

### The Insight
- Humans are incredibly data-efficient (learn from ~30M words vs 13 trillion tokens)
- Psychological frameworks compress wisdom (1K profiles > 1M random posts)
- Humans use tools (don't need to know everything, just where to find it)
- Structured reasoning might beat statistical learning for personal AI

## Proposed Approach

### Traditional Method (Baseline)
```
Medium foundation model (Mistral 7B, 32K context)
      +
Fine-tune on personal data (statistical learning)
      ↓
Personal AI
```

**Characteristics:**
- Pure neural approach
- Learn patterns from examples
- No explicit reasoning structure
- Proven to work
- Large context window (32K tokens)

### Neurosymbolic Method (Experimental)
```
Small language model (Phi-3-mini 3.8B, 4K context)
      +
Context compression layer (active + archived memory)
      +
Structured psychological framework (explicit reasoning)
      +
External memory (knowledge graph + vector store)
      +
[Optional] Tool-use layer (Track B only)
      ↓
Personal AI
```

**Characteristics:**
- Hybrid neural + symbolic
- Explicit reasoning rules
- Meta-cognitive awareness
- Context compression extends effective memory
- Unproven for this use case

**Two Tracks:**
- **Track A (Pure Local):** Phi-3 + compression + local memory, NO external calls, cost=$0
- **Track B (Hybrid, Optional):** Track A + optional tool calls with consent/caps

## What We're Testing

### Primary Hypothesis
**H1:** Neurosymbolic approach produces comparable quality to traditional fine-tuning with significantly less compute.

### Secondary Hypotheses
- **H2:** Structured psychology improves decision-making quality
- **H3:** Context compression + external memory can replace large context windows
- **H4:** Smaller model (3.8B) + structure matches medium model (7B) quality
- **H5:** Smaller model + structure is more explainable
- **H6:** Neurosymbolic approach is more data-efficient (fewer examples needed)
- **H7:** Quality data curation (Phi-3's approach) applies to personal data

## Success Criteria

### Quality Metrics
1. **Style Fidelity:** Blind Turing test ≥70% (3+ people can't distinguish from real you)
2. **Decision Accuracy:** ≥85% agreement on decision battery (30 scenarios)
3. **Factual Grounding:** <3% hallucination rate (must cite sources or admit uncertainty)

### Efficiency Metrics
1. **Training Time:** Neurosymbolic ≤50% of traditional training time
2. **Compute Cost:** Neurosymbolic ≤25% of traditional GPU-hours
3. **Data Requirements:** Neurosymbolic needs ≤50% training examples

### Explainability Metrics
1. **Reasoning Transparency:** Can trace decision path (why did it choose X?)
2. **Tool Attribution:** Can see which tool contributed what
3. **Uncertainty Awareness:** Explicitly states confidence levels

## Evaluation Design

### Comparison Framework
- Same test sets for both approaches
- Same base capabilities (both can access tools eventually)
- Same personal data sources
- Blind evaluation (evaluators don't know which is which)

### Test Batteries
1. **Communication Test:** 50 Slack-style messages, rated by friends
2. **Decision Test:** 30 realistic scenarios you'd face
3. **Knowledge Test:** 20 factual questions (some in memory, some not)
4. **Reasoning Test:** 10 multi-step problems requiring tool use

## Expected Outcomes

### If Neurosymbolic Wins
- Validates structured approach for personal AI
- Suggests path to democratization (less compute needed)
- Opens new research direction
- Document and share methodology

### If Traditional Wins
- Understand why scale matters
- Learn valuable lessons about limitations of structure
- Still have working personal AI
- Document what didn't work (negative results are valuable)

### If Tied
- Interesting: structure can match scale at lower cost
- Suggests hybrid approaches might be optimal
- Choose based on secondary factors (explainability, cost, etc)

## Risks & Limitations

### Technical Risks
- **Risk 1:** Small model lacks language competence (mitigation: Phi-3 proven capable for size)
- **Risk 2:** 4K context too limiting even with compression (mitigation: can upgrade to Mistral)
- **Risk 3:** Compression is lossy, details get lost (mitigation: store raw + compressed, retrieve when needed)
- **Risk 4:** Psychological framework too rigid (mitigation: adaptive rules)
- **Risk 5:** Local-only conflicts with tool-use vision (mitigation: Track A/B split)
- **Risk 6:** Personal data quality insufficient for small models (mitigation: semi-automated curation)

### Experimental Risks
- **Risk 4:** Evaluation bias (I built both, might favor one)
- **Risk 5:** Small sample size (just one person - me)
- **Risk 6:** Overfitting to my specific use case

### Personal Risks
- **Risk 7:** Burnout before completion (mitigation: milestone design, exit points)
- **Risk 8:** Loss of interest (mitigation: make each phase useful standalone)
- **Risk 9:** Data labeling workload too large (mitigation: semi-automated pipeline, weak labels)
- **Risk 10:** Over-engineering early phases (mitigation: defer MLOps until Phase 4+)

## Why This Matters

### Scientific Value
- Tests alternative to pure scaling paradigm
- Validates/invalidates structured reasoning for AI
- Contributes to neurosymbolic AI research

### Practical Value
- If successful, democratizes personal AI (consumer hardware viable)
- Increases diversity (everyone can have unique AI)
- Improves privacy (local training feasible)
- Enhances explainability (can see reasoning)

### Personal Value
- Learn ML end-to-end
- Build useful personal assistant
- Potentially contribute to field
- Document process for others

## What Would Invalidate This?

**Hypothesis is WRONG if:**
- Neurosymbolic quality <60% of traditional
- Training time/cost savings <10%
- System too brittle to be useful
- Psychological framework adds no value

**If wrong, what we learn:**
- Why pure neural works better
- Limitations of structured approaches
- Where to focus future efforts

## Next Steps

1. Build traditional baseline (Phase 1-2)
2. Build neurosymbolic version (Phase 3-4)
3. Run comparative evaluation (Phase 5)
4. Document results (Phase 6)
5. Share findings (blog, GitHub, possibly paper)

---

**Note:** This is a personal research project. Results may not generalize beyond my specific use case. That's okay - even narrow validation is valuable.

*Formulated: 2025-11-08*
