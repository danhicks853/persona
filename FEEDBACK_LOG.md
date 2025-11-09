# Feedback Log

**Purpose:** Public record of external feedback and how it shapes the project.  
**Goal:** Show that feedback matters and decisions evolve based on input.

All feedback is anonymized. Contributors credited as "Researcher A", "ML Engineer B", etc.

---

## Format

Each entry includes:
- **Date:** When feedback received
- **Source:** Type of contributor (anonymized)
- **Feedback Summary:** Key points raised
- **Our Response:** What we changed (or why we didn't)
- **Impact:** How this affected the project

---

## Feedback Entries

### **FB-001: Model Selection and Training Practicality**
**Date:** 2025-11-08  
**Source:** ML Engineer with fine-tuning experience  
**Status:** ✅ Accepted - Major pivot

#### **Feedback Summary**

**Key recommendations:**
1. **Use same model family for comparison** (e.g., Qwen3-1.7B vs Qwen3-8B)
   - More scientifically rigorous
   - Isolates size variable
   - Same code/architecture for both
   - Similar "mannerisms" regardless of size

2. **Consider context length capacity**
   - Warned against 4K native models (Phi-3)
   - Recommended 128K native models (Qwen)
   - Can train at 8K locally even if base is 128K
   - "You always want more context capacity"

3. **Use proven tooling (Unsloth)**
   - CUDA kernel optimizations
   - Drastically reduces memory and time
   - Can handle 8B params with QLoRA on consumer hardware
   - Link: https://unsloth.ai/

4. **Start with toy project**
   - Fine-tune small model on tiny dataset first
   - Learn pipeline before committing weeks
   - Build end-to-end with sample data to flush out bugs
   - Expect to redo everything multiple times

5. **Architecture complexity warning**
   - Custom architecture = custom inference code
   - Can spend days debugging integration
   - Avoid if possible, but acknowledged value if testing is the goal

6. **Training types clarification**
   - LM continuation training: unlabeled, next token prediction
   - Supervised fine-tuning (SFT): labeled prompt/response pairs
   - Most pipelines use SFT or both

**Quote:**
> "I get what you're trying to do and that part is a good idea, I'd like to see if it works."

#### **Our Analysis**

**What resonated:**
- Model family consistency is more rigorous scientifically
- Context capacity matters (8K > 4K headroom)
- Learn before building (toy project reduces risk)
- Proven tooling (Unsloth) over raw implementation

**What we questioned:**
- Does larger native context invalidate compression hypothesis?
- Does simplifying architecture lose the neurosymbolic core?

**Key insight:**
- We can have BOTH: Better models/tooling AND full neurosymbolic architecture
- The feedback addresses execution, not hypothesis
- Compression still valuable even with 8K base (8K compressed > 8K raw?)

#### **Decisions Made**

**✅ ACCEPTED:**

1. **Switch to Qwen family**
   - **Old:** Phi-3-mini (3.8B, 4K) vs Mistral-7B (7B, 32K)
   - **New:** Qwen3-1.7B (128K native) vs Qwen3-8B (128K native)
   - **Why:** Cleaner comparison, isolates architecture variable
   - **Trade-off:** Lose Phi-3's "textbook quality" philosophical alignment

2. **Train at 8K context** (both models)
   - **Old:** Phi-3 at 4K
   - **New:** Both Qwen models at 8K
   - **Why:** More headroom, still tests compression (8K compressed > 8K raw)
   - **Trade-off:** None - strictly better

3. **Use Unsloth for training**
   - **Old:** Raw PyTorch + Transformers
   - **New:** Unsloth framework
   - **Why:** Proven efficient, reduces memory/time
   - **Trade-off:** None - just better tooling

4. **Add toy project phase (Phase 0a)**
   - **Old:** Jump straight to full implementation
   - **New:** 2-3 day learning phase first
   - **Why:** Reduces risk, informed decisions
   - **Trade-off:** +2-3 days timeline

**❌ NOT CHANGED:**

1. **Keep full neurosymbolic architecture**
   - Context compression
   - Knowledge graph
   - Psychology framework
   - Tool use (optional)
   - **Why:** This IS the experiment - non-negotiable

2. **Keep custom wrapper layers**
   - Accept complexity warning
   - This is what we're testing
   - **Why:** "Does structure beat scale?" requires structure

3. **Keep all 7 hypotheses**
   - All remain testable with new models
   - Architecture variable is isolated better now

#### **Impact on Project**

**Timeline:**
- Added 2-3 days (toy project phase)
- Otherwise unchanged (8-12 weeks)

**Documentation updated:**
- Model selection rationale (docs/model_comparison.md)
- Model choices (SESSION_CONTEXT.md, all docs)
- Phase 0 split into 0a (toy) and 0b (full setup)
- Architecture docs unchanged

**Hypothesis strength:**
- ✅ More rigorous comparison (same family)
- ✅ Better chance of success (proven tooling)
- ✅ Lower risk (learning phase first)
- ✅ Still tests neurosymbolic fully

**Scientific validity:**
- IMPROVED: Isolating architecture variable
- IMPROVED: Fairer comparison
- MAINTAINED: All hypotheses testable

#### **Why This Is Good Feedback**

**From someone with experience:**
- Not theoretical - practical pain points
- Specific tool recommendations
- Warned about real blockers

**Improved the experiment:**
- More rigorous comparison
- Better execution strategy
- Lower risk of failure

**Respected our goals:**
- Acknowledged neurosymbolic value
- Didn't say "just use GPT-4"
- Understood we're testing, not building product

**This is the kind of feedback we want** - technical, specific, experience-based, and respectful of the experiment's goals.

---

### **FB-002: Online Learning with RLHF**
**Date:** 2025-11-08  
**Source:** ML practitioner  
**Status:** Under analysis

#### **Feedback Summary**

**Suggestion:**
> "Had an idea that might work, keeping the model in a training state while it's interacting with the user with rlhf"

**Concept:** 
- Instead of train-then-deploy, keep model trainable during inference
- Use RLHF (Reinforcement Learning from Human Feedback)
- Learn continuously from user interactions
- Model improves over time with usage

#### **Our Analysis**

**What this means:**

**Traditional approach (our current plan):**
```
Train → Freeze → Deploy → Use
                 ↑
         No further learning
```

**RLHF online learning:**
```
Train → Deploy (trainable) → Use + Learn → Improve continuously
                              ↑
                    Feedback loop active
```

**How it would work:**

**Option 1: Implicit feedback (simpler)**
```python
# During conversation
user_input = get_user_input()
response = model.generate(user_input)

# User continues conversation (implicit "this was good enough")
# OR user corrects/rephrases (implicit "that was wrong")

# Update model based on whether user accepted response
if user_accepted(response):
    reward = +1
else:
    reward = -1

# Update model weights
update_model(response, reward)  # RLHF update
```

**Option 2: Explicit feedback (better but more work)**
```python
response = model.generate(user_input)

# User rates response
feedback = get_user_rating()  # 1-5 stars or thumbs up/down

# Update model
update_model(response, reward=feedback)
```

**Option 3: Correction-based**
```python
response = model.generate(user_input)

if user_corrects(response):
    correct_response = user.provide_correct()
    # Train on correction
    fine_tune_step(user_input, correct_response)
```

#### **Implications**

**Pros:**
- ✅ Continuous improvement (learns your preferences over time)
- ✅ Adapts to changes (you evolve, model evolves with you)
- ✅ Personalization increases with usage
- ✅ Catches mistakes early (corrected immediately)
- ✅ No separate retraining needed

**Cons:**
- ❌ **Significant complexity** (reward modeling, PPO/DPO algorithms)
- ❌ **Memory overhead** (must keep gradients in memory)
- ❌ **Safety concerns** (model could drift in unexpected ways)
- ❌ **Catastrophic forgetting** (new learning overwrites old)
- ❌ **Evaluation harder** (model changing constantly)
- ❌ **Reproducibility lost** (can't replay exact model state)

**Engineering challenges:**

**1. VRAM requirements:**
```
Current (inference only):
Model (4-bit): 0.8 GB
Activations: 1-2 GB
Total: ~3-4 GB

With online training:
Model (4-bit): 0.8 GB
LoRA adapters: 0.12 GB
Activations: 2 GB
Gradients: 0.12 GB
Optimizer states: 0.24 GB
Reward model: 0.5-1 GB (if separate)
Total: ~4-5 GB

Still fits! But tighter margins.
```

**2. Algorithm complexity:**
```
Standard RLHF:
- Reward model (trained separately)
- PPO (Proximal Policy Optimization) or DPO (Direct Preference Optimization)
- KL divergence constraint (prevent drift from base model)
- Multiple update steps per interaction

Simple alternative:
- Direct fine-tuning on corrections
- Exponential moving average to prevent catastrophic forgetting
- Simpler but less theoretically grounded
```

**3. Catastrophic forgetting:**
```
Problem: 
Day 1: Learns your coding style
Day 30: You mostly chat casually
Result: Forgets how you code!

Solution:
- Replay buffer (store old examples, periodically retrain)
- EWC (Elastic Weight Consolidation) - constrain important weights
- Regular checkpoints (can roll back if drift detected)
```

**4. Feedback mechanism:**
```
Implicit: Requires sophisticated inference (did user accept response?)
Explicit: Requires UI for feedback (thumbs up/down, corrections)
Both: Add friction to user experience

For personal AI: Explicit might be okay (you're the only user)
For production: Implicit preferred (less friction)
```

#### **How It Fits Our Approach**

**Could be integrated as:**

**Phase 6+ (Future enhancement):**
- Phase 0-5: Build working system as planned
- Phase 6: Add online learning capability
- Test with simple correction-based learning first
- Evaluate if RLHF worth the complexity

**Why defer:**
- Current goal: Test neurosymbolic hypothesis
- RLHF is orthogonal (applies to both Track A and Track B)
- Adds significant complexity
- Better to prove base concept works first

**Track A + RLHF:**
```
Small model + structure + compression + online learning
Interesting combination!
Could compensate for small model size
```

**Track B + RLHF:**
```
Medium model + online learning
More standard approach
```

**Comparison still valid:**
Both could have RLHF, test which benefits more

#### **Decision**

**NOT incorporating into Phase 0-5 because:**
1. **Scope creep risk** - adds significant complexity
2. **Testing base hypothesis first** - RLHF is separate variable
3. **Can add later** - doesn't require architectural changes now
4. **Unknown benefit** - might not help small models significantly

**Could be Phase 6+ enhancement IF:**
1. Base system works (Phase 0-5 successful)
2. User wants continuous learning capability
3. Willing to accept added complexity
4. Evaluation shows it's worth the trade-offs

**Document as future consideration:**
- Add to docs/future_enhancements.md
- Not part of initial 8-12 week timeline
- Revisit after proving core hypothesis

#### **Recommendation**

**Short term (Phase 0-5):**
- Stick with offline training
- Collect feedback/corrections manually
- Periodic retraining (weekly/monthly)
- Simpler, more controllable

**Long term (Phase 6+):**
- Implement simple correction-based learning first
- Test if it improves over static model
- If successful, consider full RLHF
- Compare online vs periodic retraining

**For this feedback:**
- Acknowledge the idea (valuable concept)
- Explain why deferring (scope + hypothesis testing)
- Keep as enhancement for later phases
- Not rejecting, just prioritizing

#### **Why This Is Good Feedback**

**Shows sophisticated thinking:**
- Understands RLHF concepts
- Recognizes continuous learning value
- Relevant to personal AI domain

**Raises important questions:**
- How do we handle model drift?
- Should learning be continuous or periodic?
- What's the right feedback mechanism?

**But also:**
- Adds complexity we need to evaluate carefully
- Might be premature optimization
- Should prove base concept first

**This is exactly the kind of feedback we want** - technically sophisticated, potentially valuable, but needs careful analysis before incorporation.

---

### **FB-003: Qwen3 Model and Reasoning Data Mix**
**Date:** 2025-11-08  
**Source:** ML Engineer (Chris) with LLM deployment experience  
**Status:** Accepted - Critical methodology improvement

#### **Feedback Summary**

**Key recommendation:**
- Use **Qwen3-1.7B** instead of Qwen2.5-1.5B
- Qwen3 is newer (released 2025) with significant improvements
- Has "thinking mode" for chain-of-thought reasoning (like OpenAI's o1, but open-source)
- Better reasoning capabilities than Qwen2.5
- Superior instruction following and agent capabilities

**Critical insight discovered during implementation:**
Qwen3's thinking mode requires specific training data mix to preserve capability:
- 100% non-reasoning data → destroys thinking mode
- Need 50/50 reasoning/non-reasoning mix (per Unsloth docs)
- For psychology capture, we NEED reasoning mode (it's how you think!)

#### **Our Analysis**

**Why this matters for our project:**

1. **Psychology = Reasoning Process**
   - We're not just cloning writing style
   - We're capturing HOW the user thinks
   - Mental models, heuristics, decision-making patterns
   - Thinking mode is PERFECT for this

2. **Hypothesis Alignment**
   - H2: "Structured psychology improves decision accuracy"
   - Psychology framework = reasoning patterns, not just outputs
   - Need to capture the PROCESS, not just the conclusions

3. **Data Collection Impact**
   - Originally planned: 20 style, 20 facts, 20 decisions (all direct answers)
   - Updated plan: 15 style, 15 facts, 30 decisions (with chain-of-thought)
   - 50% reasoning examples preserve thinking mode capability
   - Decisions must show `<thinking>` tags with analysis

#### **Changes Made**

**Model Switch:**
```python
# Old
model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # 1.5B, 128K context

# New  
model_name = "unsloth/Qwen3-1.7B"  # 1.7B, 32K context, thinking mode
```

**Dataset Composition Updated:**
- **15 Style examples** - NON-REASONING (direct responses)
- **15 Fact examples** - NON-REASONING (factual answers)
- **30 Decision/Psychology examples** - REASONING (chain-of-thought with `<thinking>` tags)

**Reasoning Example Format:**
```json
{
  "id": "decision_001",
  "messages": [
    {
      "role": "user",
      "content": "Should we automate this or do it manually?"
    },
    {
      "role": "assistant", 
      "content": "<thinking>\nFactors:\n- Deadline: 2 days\n- Automation: 10 hours\n- Manual: 4 hours\n\nHeuristics:\n- Accuracy > Speed\n- Don't let perfect be enemy of done\n\nDecision: Manual now, automate after\n</thinking>\n\nManual for now. Here's why: [explanation]"
    }
  ],
  "metadata": {
    "category": "decisions",
    "reasoning": "true"
  }
}
```

**Documentation Updated:**
- `docs/phases/phase0a.md` - Qwen3, 50/50 mix
- `docs/DATA_COLLECTION_GUIDE.md` - Reasoning examples with `<thinking>` tags
- `SESSION_CONTEXT.md` - Model choice and dataset requirements
- `data/phase0a/README.md` - Templates for both formats
- `tests/model_loading/test_qwen_load.py` - Tests Qwen3

#### **Validation**

**Tested Qwen3-1.7B:**
- Loaded successfully with 4-bit quantization
- VRAM: 1.42GB (even better than Qwen2.5's 1.55GB)
- Inference working
- 14.5GB VRAM still available for training
- Unsloth has pre-optimized version

**Why Qwen3 is Better:**
1. **Thinking mode** - Can show reasoning process (critical for psychology)
2. **Better reasoning** - Improved over Qwen2.5 in math/code/logic
3. **Better instruction following** - More aligned with human preferences
4. **Better agent capabilities** - Tool use and complex tasks
5. **Latest generation** - Most current model (future-proof)

**Trade-offs Accepted:**
- 32K context vs 128K (but training at 8K anyway, so no practical impact)
- Slightly newer/less battle-tested (but official Unsloth support mitigates)

#### **Impact on Project**

**Immediate (Phase 0a):**
- Changed data collection approach entirely
- 30 examples must show reasoning process with `<thinking>` tags
- More time to collect (2-3 hours vs 1-2 hours)
- Better alignment with psychology capture goal

**Long-term (Full project):**
- Thinking mode enables better psychology modeling
- Can capture decision-making heuristics explicitly
- Chain-of-thought data shows mental models in action
- More rigorous test of "small + structure" hypothesis

**Methodological Improvement:**
- Caught a critical gap: training data format affects capability preservation
- Learned about reasoning vs non-reasoning data balance
- Unsloth recommends 75/25 reasoning/non-reasoning for Qwen3
- We're doing 50/50 as compromise (preserve thinking, learn style/facts)

#### **Why This Is Excellent Feedback**

**Technical correctness:**
- Qwen3 IS better than Qwen2.5 (objectively newer, improved)
- Thinking mode IS valuable for our use case
- Data mix requirement IS documented by Unsloth

**Perfect timing:**
- Before any training (zero cost to switch)
- Before data collection (can adjust approach)
- During Phase 0a (testing phase specifically for this)

**Aligns with hypothesis:**
- Psychology = reasoning patterns
- Thinking mode captures this perfectly
- Better tool for the actual goal

**Caught a gap we missed:**
- We didn't know about reasoning/non-reasoning data mix requirement
- Would have destroyed thinking mode with our original plan
- Now collecting the RIGHT kind of data for the capability we need

#### **Lessons Learned**

1. **Latest isn't always in training data** - Qwen3 released after most LLM knowledge cutoffs
2. **Community knowledge matters** - Chris knew about Qwen3, we didn't
3. **Test assumptions early** - Phase 0a caught this before full investment
4. **Read the docs carefully** - Unsloth docs had reasoning mix requirement
5. **Preserve model capabilities** - Training data format affects what model can do

**This feedback fundamentally improved the methodology. Thank you, Chris!**

---

### **FB-004: [Template]**
**Date:** TBD  
**Source:** TBD  
**Status:** Pending

---
### **FB-005: [Future feedback]**
**Date:** TBD  
**Source:** TBD  
**Status:** Pending

---

## Feedback We're Looking For

**Technical:**
- Flaws in hypothesis or methodology
- Missing evaluation criteria
- Better ways to test the same thing
- Tools/techniques we should know about

**Practical:**
- "I tried this and here's what I learned"
- Common pitfalls to avoid
- Realistic timeline estimates
- Resource requirements

**Scope:**
- Is this too ambitious?
- Is this too narrow?
- What should be added/removed?

**Value:**
- If it works, is it useful?
- If it fails, are the learnings valuable?
- Worth the time investment?

---

## Feedback We're NOT Looking For

**Dismissive:**
- ❌ "Just use GPT-4" (not testing scale)
- ❌ "This is impossible" (without reasoning)
- ❌ "Why bother?" (democratization matters)

**Off-scope:**
- ❌ "Build a product" (not the goal)
- ❌ "Get a team" (testing solo feasibility)
- ❌ "This needs $5M" (consumer hardware constraint)

**Unconstructive:**
- ❌ Vague concerns without specifics
- ❌ "I don't like X" without alternatives
- ❌ Gatekeeping ("you need a PhD")

---

## How to Submit Feedback

**GitHub:**
- Open an issue: https://github.com/danhicks853/persona/issues
- Tag with `feedback` label

**Direct:**
- Text, Facebook, or Github Issue. If I've directed you here, you know at least one of those methods ;)

**What helps:**
- Specific technical points
- Experience-based insights
- Constructive alternatives
- References to papers/tools

**Your feedback shapes this project. We listen and adapt.**

---

## Summary Statistics

**Total feedback received:** 3  
**Major pivots:** 2 (FB-001, FB-003)  
**Deferred for later:** 1 (FB-002)  
**Minor adjustments:** 0  
**Feedback declined:** 0  

**Most impactful:** 
- FB-001: Model selection and Unsloth framework
- FB-003: Qwen3 and reasoning data methodology

---

*Last updated: 2025-11-08, 10:10pm*
