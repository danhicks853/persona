# Phase 0a: Toy Project - Learn the Pipeline

**Goal:** Hands-on learning with Unsloth before committing to full build. Test compression concepts early.

**Duration:** 3 days (time-boxed)

**Status:** Not started

---

## Overview

### **What We're Building**

A minimal working version of the full system:
1. Fine-tune Qwen3-1.7B on 60 real examples (reasoning + non-reasoning mix)
2. Test basic inference (including thinking mode)
3. Implement simple compression layer
4. Validate the approach works

**Model Choice:** Qwen3-1.7B (latest generation, supports reasoning mode)

### **What We're Learning**

**Technical:**
- How Unsloth fine-tuning works
- Memory requirements (actual VRAM usage)
- Training time (real-world estimates)
- Inference speed
- Where things break

**Strategic:**
- Is the data format workable?
- Does compression concept hold up?
- Are we missing something obvious?
- Should we adjust before full build?

### **Exit Criteria**

**✅ Proceed to full build if:**
- Model trains successfully
- Inference works and makes sense
- Simple compression doesn't lose critical info
- VRAM usage acceptable (<15GB training, <8GB inference)
- No major architectural flaws discovered

**🔄 Adjust plan if:**
- Training takes way longer than expected
- VRAM constraints tighter than planned
- Data format needs changes
- Compression too lossy (but fixable)

**❌ Pivot/stop if:**
- Can't train on hardware (VRAM insufficient)
- Fundamental architecture flaw discovered
- Compression completely fails
- Would take >12 weeks to complete full project

---

## Day 1: Setup + Data Collection

### **Morning: Environment Setup (2-3 hours)**

**Tasks:**
1. Create conda environment
2. Install PyTorch + CUDA
3. Install Unsloth
4. Test GPU access
5. Download Qwen-1.5B

**Checklist:**
- [ ] `conda create -n persona python=3.10`
- [ ] `conda activate persona`
- [ ] `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
- [ ] `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`
- [ ] Test: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Test: `python -c "from unsloth import FastLanguageModel; print('OK')"`
- [ ] Quick load test (model loads, check VRAM)

**Expected VRAM (4-bit):**
- Model loading: ~2-3GB
- Inference: ~4-6GB
- Training: ~10-14GB with 8K context

**If issues:**
- Check CUDA version matches PyTorch
- Try `nvidia-smi` to verify GPU
- Check Unsloth GitHub issues

### **Afternoon: Data Collection (2-4 hours)**

**Goal:** 60 real examples with reasoning/non-reasoning mix

**Dataset composition (50/50 mix to preserve Qwen3 thinking mode):**
- **15 style examples** - NON-REASONING (how you write/respond directly)
- **15 fact examples** - NON-REASONING (factual info about you)
- **30 decision/psychology examples** - REASONING (chain-of-thought, show your thinking process)

**Why this mix?**
- Qwen3 has "thinking mode" for reasoning tasks
- Training on 100% non-reasoning data would destroy this capability
- Psychology = capturing HOW you think, not just WHAT you say
- 50/50 mix preserves reasoning while learning your style/facts

**Sources:**
- Slack DMs/messages (export recent conversations)
- Discord chats (personal servers)
- Email responses (professional writing)
- GitHub comments/commits (technical writing)
- Ticket responses (work style)
- Personal notes/journal (if available)

**Collection process:**
1. Export raw text from sources
2. Format into conversation pairs
3. Categorize by type (style/fact/decision)
4. Save as JSONL (see data_format.md)
5. Split: 50 train, 10 test

**Format example:**
```json
{
  "messages": [
    {"role": "user", "content": "What's your take on this bug?"},
    {"role": "assistant", "content": "Looking at the stack trace, it's a null pointer in the auth middleware. I'd add defensive checks and log the request context so we can debug if it happens again. Quick fix is wrap it in try-catch, proper fix is validate tokens upstream."}
  ],
  "metadata": {
    "category": "style",
    "source": "slack",
    "context": "work_technical"
  }
}
```

**Quick shortcuts:**
- Don't over-curate (real data, warts and all)
- 1-2 turn conversations (keep simple)
- Aim for diversity (different topics)

**Deliverable:**
- `data/toy/train.jsonl` (50 examples)
- `data/toy/test.jsonl` (10 examples)

---

## Day 2: Training + Basic Inference

### **Morning: First Training Run (2-3 hours)**

**Setup script:** `scripts/toy_train.py`

**Training config:**
```python
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=2048,  # Start small for toy
    dtype=None,
    load_in_4bit=True,
)

# Prepare LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Training args
training_args = TrainingArguments(
    output_dir="models/toy/checkpoints",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=100,  # Quick toy run
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",  # Formatted prompt
    max_seq_length=2048,
    args=training_args,
)

trainer.train()
```

**What to monitor:**
- VRAM usage (log at each step)
- Training loss (should decrease)
- Time per step (estimate total time)
- Any errors/warnings

**Expected:**
- ~10-15 minutes for 100 steps
- Loss starts ~2-3, ends ~0.5-1.0
- VRAM peaks ~12-14GB

**Save:**
- Model checkpoint: `models/toy/final`
- Training log: `models/toy/training.log`

### **Afternoon: Basic Inference Testing (2 hours)**

**Test script:** `scripts/toy_test.py`

**Tests to run:**

**1. Exact recall (memorization check):**
```python
# Pick 3 training examples
# Feed same prompt, see if similar response
```

**2. Generalization (unseen prompts):**
```python
# Test set (10 held-out examples)
# Check if responses make sense
```

**3. Style check:**
```python
# Generic prompt: "How would you debug a memory leak?"
# Does it sound like you?
```

**4. Fact check:**
```python
# "What do you do for work?"
# Does it know facts about you?
```

**Evaluation:**
- Manual review (read responses)
- Note: coherent? on-topic? in your style?
- Red flags: gibberish, hallucinations, off-style

**Deliverable:**
- `models/toy/test_results.txt` (responses + notes)

---

## Day 3: Compression + Evaluation

### **Morning: Simple Compression Layer (3 hours)**

**Goal:** Test if compression concept works at all.

**Implementation:** `scripts/toy_compress.py`

**Simple approach:**
```python
# Simulate multi-turn conversation
conversation = [
    {"role": "user", "content": "Turn 1 question"},
    {"role": "assistant", "content": "Turn 1 response"},
    {"role": "user", "content": "Turn 2 question"},
    {"role": "assistant", "content": "Turn 2 response"},
    {"role": "user", "content": "Turn 3 question"},
    {"role": "assistant", "content": "Turn 3 response"},
    {"role": "user", "content": "Turn 4 question (references turn 1)"},
]

# Compress turns 1-2 to summary
summary = model.generate(
    "Summarize this conversation in 2 sentences:\n" + turns_1_2
)

# New context: summary + turns 3-4 + current question
compressed_context = [
    {"role": "system", "content": f"Previous context: {summary}"},
    *turns_3_4,
    current_question
]

# Does the response still make sense?
response = model.generate(compressed_context)
```

**Test questions:**
1. Can it summarize accurately?
2. Does compressed context preserve key info?
3. Does response quality degrade significantly?

**Metrics:**
- Token count: original vs compressed (should be ~50% smaller)
- Response quality: with vs without compression (manual eval)
- Info loss: can it still answer questions about turn 1?

**Expected:**
- Some info loss acceptable
- Should maintain gist
- 30-50% compression ratio

### **Afternoon: Analysis + Decision (2 hours)**

**Review:**
1. Training: smooth? issues?
2. Inference: quality acceptable?
3. Compression: concept viable?
4. VRAM: within limits?
5. Time: realistic for full project?

**Document findings:**
- `docs/toy_project_report.md`
- What worked
- What didn't
- What to adjust
- Proceed or pivot?

**Decision tree:**

**PROCEED if:**
- ✅ Model trained successfully
- ✅ Responses coherent and on-topic
- ✅ Compression reduces tokens without breaking
- ✅ VRAM <15GB training, <8GB inference
- ✅ No showstoppers discovered

**ADJUST if:**
- 🔄 Training slow but workable (adjust batch size)
- 🔄 Compression lossy but improvable (better prompts)
- 🔄 VRAM tight but manageable (smaller batches)
- 🔄 Minor issues (fixable)

**PIVOT if:**
- ❌ Can't train on hardware
- ❌ Compression completely fails
- ❌ Model quality terrible even on toy set
- ❌ Would take >12 weeks for full project
- ❌ Fundamental architecture flaw

---

## Deliverables

**Code:**
- [ ] `scripts/toy_train.py` - Training script
- [ ] `scripts/toy_test.py` - Inference testing
- [ ] `scripts/toy_compress.py` - Compression test
- [ ] `data/toy/train.jsonl` - Training data (50 examples)
- [ ] `data/toy/test.jsonl` - Test data (10 examples)

**Models:**
- [ ] `models/toy/final/` - Trained checkpoint
- [ ] `models/toy/training.log` - Training metrics

**Documentation:**
- [ ] `docs/toy_project_report.md` - Findings and decision
- [ ] `models/toy/test_results.txt` - Inference examples
- [ ] Update SESSION_CONTEXT.md with learnings

---

## Common Issues and Solutions

### **VRAM Errors**

**Symptom:** OOM during training

**Solutions:**
- Reduce batch size (try 1)
- Increase gradient accumulation
- Reduce max_seq_length (try 1024)
- Use more aggressive quantization

### **Slow Training**

**Symptom:** >1 hour for 100 steps

**Solutions:**
- Check GPU utilization (`nvidia-smi`)
- Verify CUDA installed correctly
- Reduce sequence length
- Try different optimizer

### **Poor Quality Responses**

**Symptom:** Gibberish or off-topic

**Solutions:**
- Check data format (might be corrupted)
- Try more training steps (100 might be too few)
- Adjust learning rate (try 3e-4)
- Verify tokenizer working correctly

### **Compression Fails**

**Symptom:** Summaries useless or wrong

**Solutions:**
- Try different summarization prompt
- Use external summarizer (not fine-tuned model)
- Simplify compression test
- This is OK - full project can use better approach

---

## Success Criteria

**Minimum viable outcome:**
- ✅ Successfully trained a model
- ✅ Model produces coherent responses
- ✅ Learned how Unsloth works
- ✅ Identified at least 3 gotchas/learnings
- ✅ Can estimate realistic timeline for full project

**Ideal outcome:**
- ✅ All minimum criteria
- ✅ Compression test shows promise
- ✅ VRAM usage comfortable (<12GB training)
- ✅ Training time fast (<30 mins for toy set)
- ✅ Responses clearly in your style
- ✅ Confident to proceed with full build

**Even if compression test fails or quality isn't perfect, if we learned the pipeline and didn't hit showstoppers, that's success.**

---

## After Phase 0a

**If proceeding:**
- Update SESSION_CONTEXT.md with learnings
- Refine Phase 1 plan based on discoveries
- Commit toy project code and findings
- Start full data collection

**If pivoting:**
- Document what didn't work
- Update FEEDBACK_LOG.md with self-feedback
- Discuss alternatives
- Decide: adjust or pause?

---

**Time-boxed at 3 days. Goal is learning, not perfection.**

*Remember: Even "failure" produces valuable data.*
