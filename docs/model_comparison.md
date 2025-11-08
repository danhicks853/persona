# Model Comparison & Selection

**Decision Date:** 2025-11-08 (Morning session)

---

## Models Considered

### **Phi-3-mini-4k-instruct** (CHOSEN for Track A)
- **Developer:** Microsoft Research
- **Size:** 3.8B parameters
- **Context:** 4K tokens (~3,000 words)
- **Training philosophy:** Quality > Quantity (curated textbooks, reasoning examples)
- **Gated:** No (immediate download)
- **License:** MIT

### **Mistral-7B-Instruct-v0.2** (CHOSEN for Track B)
- **Developer:** Mistral AI
- **Size:** 7B parameters
- **Context:** 32K tokens (~24,000 words)
- **Training philosophy:** Efficient architecture (sliding window attention)
- **Gated:** No (immediate download)
- **License:** Apache 2.0

### **Llama-3.1-8B-Instruct** (Considered but not chosen)
- **Developer:** Meta
- **Size:** 8B parameters
- **Context:** 128K tokens
- **Gated:** Yes (requires approval)
- **Why not chosen:** Gating adds friction, Mistral sufficient

---

## What Makes Models Different?

### **1. Size (Parameters)**

**What it means:** Number of learnable weights

**Impact:**
- More params = more capacity for knowledge and reasoning
- More params = more VRAM, slower inference
- Diminishing returns (70B isn't 10x better than 7B)

**Examples:**
- Phi-3: 3.8B
- Mistral: 7B
- Llama-3.1: 8B, 70B, 405B variants

**For our use:** 3.8B-8B is the sweet spot (fits hardware, good capability)

---

### **2. Training Data Quality**

**What it means:** What text the model learned from

**Phi-3's innovation:**
- Small dataset, HIGH quality (textbooks, reasoning examples)
- Shows data quality > data quantity
- 3.8B Phi-3 beats many 7B models on benchmarks

**Why this matters for us:**
> Your thesis: "Structure + quality data > scale"
> Phi-3: Proof that thesis can work
> Project goal: Apply Phi-3's philosophy to personal AI

---

### **3. Context Length**

**What it is:** How many tokens the model can "see" at once

**What determines it:**
1. **Positional encoding size** (baked into architecture during pre-training)
2. **Memory budget** (O(n²) attention cost)
3. **Training compute** (longer context = more expensive)

**Examples:**
- Phi-3: 4K (design trade-off: spent budget on quality, not length)
- Mistral: 32K (efficient sliding window attention)
- Llama-3.1: 128K (massive context)

**Trade-offs:**
- Small context = less memory, faster, but limited history
- Large context = remembers more, but expensive, slower

**Our solution:** Phi-3's 4K + compression = effective infinite context

---

### **4. Instruction Tuning**

**What it is:** Post-training to follow instructions

**Without instruction tuning:**
```
User: "Write me a function"
Model: "to write a function you need to use def keyword..."
```
(Continues text like autocomplete)

**With instruction tuning:**
```
User: "Write me a function"
Model: "def example(param):
    return result"
```
(Actually follows the instruction)

**All three models we considered are instruction-tuned** (have "-Instruct" suffix)

---

### **5. Specialization**

**General models:**
- Phi-3-Instruct, Mistral-Instruct, Llama-Instruct
- Good at conversation, reasoning, general tasks

**Specialized models:**
- CodeLlama (code-focused)
- Minerva (math-focused)
- StarCoder (programming)

**For personal AI:** General models are best (need versatility)

---

## Benchmark Comparison

### **Common Benchmarks**

**MMLU (Knowledge):** Tests general knowledge (history, science, etc)
- Phi-3: ~69%
- Mistral-7B: ~60%
- Llama-3.1-8B: ~66%
- **Winner: Phi-3** (despite being smallest!)

**HumanEval (Code):** Python function writing
- Phi-3: ~50%
- Mistral-7B: ~30%
- Llama-3.1-8B: ~62%
- **Winner: Llama-3.1**

**GSM8K (Math):** Grade school math word problems
- Phi-3: ~85%
- Mistral-7B: ~50%
- Llama-3.1-8B: ~80%
- **Winner: Phi-3**

**MT-Bench (Conversation):** Multi-turn chat quality
- Phi-3: 7.8/10
- Mistral-7B: 7.6/10
- Llama-3.1-8B: 8.0/10
- **Winner: Llama-3.1** (marginally)

### **What This Means**

**Phi-3 punches WAY above its weight:**
- 3.8B params beating 7-8B models
- Validates "quality data > quantity" thesis
- Proves small models can be very capable

**But benchmarks don't tell whole story:**
- Fine-tunability matters (not benchmarked)
- Personal AI needs adaptability, not just raw scores
- Real-world use differs from academic tests

---

## Why We Chose Phi-3 for Track A

### **Philosophical Alignment**

**Your thesis:**
> "What if we don't need billions of parameters? What if structure + psychology + tools > brute force?"

**Phi-3's thesis:**
> "3.8B params + curated textbooks > 7B params + web scrapes"

**Perfect match.** If you succeed with Phi-3, you've proven BOTH theses.

### **Practical Advantages**

**Speed:**
- Faster download (smaller)
- Faster training (fewer params)
- Faster inference (less computation)
- More iterations per day

**Efficiency:**
- Uses less VRAM (more headroom for experiments)
- Lower power consumption
- Can run on more hardware

**Validation:**
- If small model + structure works → strong result
- If you need Mistral → learn why context/params matter
- Either way, better understanding

### **Cost = $0**

- No gating (unlike Llama)
- No API costs (local-only Track A)
- No waiting for approval

---

## Why We Chose Mistral for Track B

### **Complementary Design**

**Mistral represents the "traditional" approach:**
- Medium size (7B)
- Large context (32K)
- Proven fine-tuning success
- Strong community support

**Perfect baseline** to compare against Phi-3+structure

### **Technical Advantages**

**Context window:**
- 32K tokens = ~24,000 words
- Can hold long conversations without compression
- Tests if compression is necessary or just a workaround

**Architecture:**
- Sliding window attention (efficient)
- Grouped-query attention (fast)
- Well-optimized for inference

**Community:**
- Lots of fine-tuning examples
- Known hyperparameters
- Documented best practices

### **Not Gated**

- Download immediately
- No approval needed
- No terms-of-service restrictions

---

## Context Length Deep Dive

### **What 4K vs 32K Actually Means**

**4K tokens (Phi-3):**
- ~10-15 conversation turns
- Short documents
- Single script with context
- **Sufficient for:** Most personal interactions, focused tasks
- **Insufficient for:** Very long conversations, entire codebases, meeting transcripts

**32K tokens (Mistral):**
- ~80-120 conversation turns
- Long documents, multiple files
- Extensive context
- **Sufficient for:** Almost everything
- **Overkill for:** Most tasks

### **Our Context Compression Solution**

**Instead of 32K context, use 4K + compression:**

```
Active context (4K):
  - Psychology profile (500t, always loaded)
  - Last 5 turns (1500t, verbatim)
  - Retrieved memories (1000t, from compression)
  - Current task (500t)
  - Available (500t for tool outputs)

Compressed memory (unlimited):
  - Old conversation summaries
  - Extracted decisions/facts
  - Pattern library

Raw archive (unlimited):
  - Full fidelity storage
  - Retrievable when needed
```

**Result:** Effective "infinite context" with 4K window

**This tests hypothesis H3:** "Context compression + external memory can replace large context windows"

---

## The Experiment Design

### **Track A (Neurosymbolic)**
```
Phi-3-mini (3.8B, 4K)
+ Context compression
+ External memory (knowledge graph)
+ Structured psychology
+ NO external tools (local-only)
```

**Goal:** Prove small + structure ≈ medium model

### **Track B (Traditional)**
```
Mistral-7B (7B, 32K)
+ Traditional fine-tuning
+ All context in-window
+ (Optional) Add compression for fair comparison
```

**Goal:** Establish baseline, see if Phi-3 can match

### **Comparison Metrics**

**Quality:**
- Style Turing test (which sounds like Dan?)
- Decision accuracy (would Dan do this?)
- Hallucination rate (makes up facts?)

**Efficiency:**
- Training time (hours)
- GPU-hours (compute cost)
- Inference speed (tokens/sec)
- VRAM usage (GB)

**Explainability:**
- Can you trace reasoning?
- Can you see what was retrieved?
- Does it state uncertainty?

---

## Can We Add Mistral Later?

**Yes, the architecture supports both:**

```python
# Configuration-driven model selection
config = {
    "track_a": {
        "model": "microsoft/Phi-3-mini-4k-instruct",
        "context": 4096,
        "compression": True,
        "external_memory": True
    },
    "track_b": {
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "context": 32768,
        "compression": False,  # Optional
        "external_memory": False  # Test raw first
    }
}
```

**Upgrade path:**
1. Start with Phi-3 (Phase 0-2)
2. Validate compression works (Phase 3)
3. Add Mistral for comparison (Phase 4-5)
4. Run A/B evaluation (Phase 5)

**If Phi-3 + compression proves insufficient:**
- Add Mistral mid-project
- Keep Phi-3 learnings
- Document why upgrade was needed

---

## Gated Models Explained

### **What "Gated" Means**

**Gated model:** Requires explicit permission before download

**Process:**
1. Visit model page on Hugging Face
2. Click "Agree to Terms" or "Request Access"
3. Wait for approval (instant to days)
4. Login with token to download

**Examples:**
- ✅ Gated: Llama 3.1, Gemma
- ❌ Not gated: Phi-3, Mistral

### **Why Models Are Gated**

**Reasons:**
- License agreements (need to accept terms)
- Usage tracking (company wants to know who's using)
- Safety concerns (powerful models)
- Research restrictions (academic use only)
- Export control (some countries blocked)

### **Why We Avoided Gated Models**

**Friction:**
- Adds delay (approval takes time)
- Extra steps (authentication, tokens)
- Possible rejection (if terms not acceptable)

**Philosophy:**
- Project is about democratization
- Gating is antithetical to "everyday dude" goal
- Want immediate access, no barriers

**Practical:**
- Phi-3 and Mistral are sufficient
- No need to wait for Llama approval
- Can start immediately

---

## Model Selection Checklist

When choosing a model for future experiments:

**Size:**
- [ ] Fits in 20GB VRAM with 4-bit quantization?
- [ ] Small enough for fast iteration?
- [ ] Large enough for task complexity?

**Context:**
- [ ] 4K sufficient with compression, or need more?
- [ ] Can afford the VRAM for longer context?

**Availability:**
- [ ] Not gated (or willing to wait)?
- [ ] Open weights (can fine-tune)?
- [ ] License allows use case?

**Performance:**
- [ ] Good benchmark scores for task type?
- [ ] Known fine-tuning success stories?
- [ ] Community support and examples?

**Philosophy:**
- [ ] Aligns with project goals?
- [ ] Tests a specific hypothesis?

---

## Summary

**Chosen Models:**
- **Track A:** Phi-3-mini (3.8B, 4K) - neurosymbolic approach
- **Track B:** Mistral-7B (7B, 32K) - traditional baseline

**Rationale:**
- Phi-3 aligns philosophically (quality > scale)
- Mistral provides strong baseline
- Both not gated (immediate start)
- Different enough to test hypotheses
- Similar enough for fair comparison

**Key Innovation:**
- Context compression system
- Tests if small+structure can match medium+scale
- Validates "everyday dude" can build personal AI

**Next Steps:**
- Download Phi-3-mini (Phase 0)
- Validate hardware/setup works
- Later add Mistral for comparison

---

*Model comparison and selection - 2025-11-08*
