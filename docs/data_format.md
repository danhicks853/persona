# Data Format Specification

**Purpose:** Standard format for all training data across the project.

**Updated:** 2025-11-08

---

## Overview

### **File Format**

**JSONL (JSON Lines):**
- One JSON object per line
- Easy to stream
- Easy to append
- Human-readable

**Directory structure:**
```
data/
├── toy/              # Phase 0a (50-60 examples)
│   ├── train.jsonl
│   └── test.jsonl
├── phase1/           # Knowledge/facts (500-1K examples)
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── phase2/           # Style (1K-2K examples)
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── phase3/           # Psychology (500-1K examples)
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
└── full/             # Combined dataset
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl
```

---

## Standard Format

### **Conversation Format (Primary)**

```json
{
  "id": "unique_identifier",
  "messages": [
    {
      "role": "user",
      "content": "User message or question"
    },
    {
      "role": "assistant",
      "content": "Your response"
    }
  ],
  "metadata": {
    "category": "style|facts|decisions|psychology",
    "source": "slack|discord|email|github|manual",
    "timestamp": "2024-11-08T12:00:00Z",
    "context": "work|personal|technical|casual",
    "quality": "high|medium|low",
    "tokens": 150,
    "notes": "Optional annotation"
  }
}
```

### **Multi-turn Format**

```json
{
  "id": "conv_001",
  "messages": [
    {"role": "user", "content": "Question 1"},
    {"role": "assistant", "content": "Response 1"},
    {"role": "user", "content": "Follow-up referencing response 1"},
    {"role": "assistant", "content": "Response 2"}
  ],
  "metadata": {
    "category": "decisions",
    "turns": 2,
    "requires_context": true
  }
}
```

---

## Category Specifications

### **1. Style Examples**

**Purpose:** Capture writing style, tone, mannerisms

**Content:**
- How you respond to questions
- Email/message style
- Technical explanations
- Casual conversation

**Sources:**
- Slack DMs
- Discord messages
- Email responses
- GitHub comments
- Code review comments

**Example:**
```json
{
  "id": "style_001",
  "messages": [
    {
      "role": "user",
      "content": "The deployment failed with a 500 error. Any ideas?"
    },
    {
      "role": "assistant",
      "content": "Check the logs first - usually it's a config issue or missing env var. If it's the auth service timing out, we might need to bump the connection pool. Can you paste the error stack?"
    }
  ],
  "metadata": {
    "category": "style",
    "source": "slack",
    "context": "work_technical",
    "quality": "high"
  }
}
```

### **2. Fact Examples**

**Purpose:** Teach factual knowledge about you

**Content:**
- Job/role details
- Technical skills
- Projects you've worked on
- Tools you use
- Personal info (hobbies, preferences)

**Format variations:**

**Direct Q&A:**
```json
{
  "id": "fact_001",
  "messages": [
    {"role": "user", "content": "What do you do for work?"},
    {"role": "assistant", "content": "I'm an automation engineer at an MSP. I work on RMM migrations, PowerShell scripting, and building internal tools like HelixChat and HelixIQ."}
  ],
  "metadata": {"category": "facts", "subcategory": "work"}
}
```

**Contextual knowledge:**
```json
{
  "id": "fact_002",
  "messages": [
    {"role": "user", "content": "How would you approach this Kaseya VSA migration?"},
    {"role": "assistant", "content": "Based on my experience with the HelixAPI integration, I'd start by mapping the VSA procedures to our automation framework, then batch-migrate procedures with proper error handling."}
  ],
  "metadata": {"category": "facts", "subcategory": "experience"}
}
```

### **3. Decision Examples**

**Purpose:** Model decision-making patterns and reasoning

**Content:**
- Technical decisions
- Prioritization choices
- Trade-off reasoning
- Risk assessment
- "What would you do" scenarios

**Example:**
```json
{
  "id": "decision_001",
  "messages": [
    {
      "role": "user",
      "content": "We can either refactor this now (2 weeks) or ship with tech debt and refactor later. Client wants features ASAP."
    },
    {
      "role": "assistant",
      "content": "Ship with debt if it's isolated and documented. Refactor blocks other features, and client pressure is real. But: set a hard deadline for cleanup (2 sprints max), add clear TODOs, and make sure it won't cascade. If the debt will spread or block future work, refactor now even if it hurts."
    }
  ],
  "metadata": {
    "category": "decisions",
    "subcategory": "technical_tradeoffs",
    "values_demonstrated": ["pragmatism", "documentation", "long_term_thinking"]
  }
}
```

### **4. Psychology Examples (Phase 3+)**

**Purpose:** Model trait patterns, trauma responses, cognitive biases

**Content:**
- How you react under stress
- Triggers and coping mechanisms
- Value hierarchies in action
- Decision heuristics
- Conflict resolution patterns

**Example:**
```json
{
  "id": "psych_001",
  "messages": [
    {
      "role": "user",
      "content": "This critical bug was just found in production. Client is escalating. What's your first move?"
    },
    {
      "role": "assistant",
      "content": "First: breathe and don't panic-code. Check if anyone's actively impacted right now. If yes, immediate mitigation (rollback or workaround). If no, I have time to think. Then: reproduce the bug, understand root cause, fix properly. Under pressure I default to precision over speed - better to take 30 mins and fix it right than rush a patch that makes it worse."
    }
  ],
  "metadata": {
    "category": "psychology",
    "traits_demonstrated": ["stress_response", "precision_over_speed", "systematic_thinking"],
    "coping_mechanism": "pause_and_assess"
  }
}
```

---

## Data Quality Guidelines

### **High Quality Indicators**

**Content:**
- ✅ Authentic (actually you, not "how you should sound")
- ✅ Specific (concrete details, not generic)
- ✅ Complete (full thoughts, not fragments)
- ✅ Natural (conversational, not formal unless you're formal)

**Format:**
- ✅ Valid JSON
- ✅ All required fields present
- ✅ Metadata accurate
- ✅ Reasonable length (50-500 tokens per message)

### **Low Quality Indicators**

**Content:**
- ❌ Generic (could be anyone)
- ❌ Vague (no specifics)
- ❌ Incomplete (fragments)
- ❌ Inconsistent (contradicts other examples)

**Format:**
- ❌ Malformed JSON
- ❌ Missing fields
- ❌ Wrong category
- ❌ Too long (>1000 tokens) or too short (<10 tokens)

### **Quality Levels**

**High (keep all):**
- Perfect examples of your patterns
- Rich context
- Clear demonstration of style/knowledge/reasoning

**Medium (keep most):**
- Good but not perfect
- Some missing context but usable
- Might need light editing

**Low (filter or fix):**
- Generic or vague
- Needs significant editing
- Consider excluding from training

---

## Data Collection Workflow

### **Phase 0a (Toy Project)**

**Target:** 50 train, 10 test

**Process:**
1. Export raw conversations from sources
2. Select diverse examples
3. Format to JSONL
4. Add metadata
5. Manual quality check
6. Split train/test (don't leak!)

**Speed over perfection:**
- Don't over-curate for toy project
- Real data with warts is fine
- Goal is learning, not perfect dataset

### **Phase 1+ (Full Project)**

**Target:** 2K-5K total examples

**Process:**
1. Bulk export from all sources
2. Automated weak labeling (category detection)
3. Spot-check 10-20% manually
4. Format and standardize
5. Quality filter
6. Deduplication
7. Train/val/test split (80/10/10)

**Semi-automated pipeline:**
- Scripts to parse common formats
- Category detection heuristics
- Automated quality scoring
- Manual validation on sample

---

## Special Cases

### **System Instructions**

For fine-tuning with system prompts:

```json
{
  "id": "sys_001",
  "messages": [
    {
      "role": "system",
      "content": "You are a personal AI assistant trained on one person's patterns. Be direct, technically accurate, and acknowledge uncertainty when you don't know something."
    },
    {"role": "user", "content": "Question"},
    {"role": "assistant", "content": "Response"}
  ],
  "metadata": {"has_system_prompt": true}
}
```

### **Context Compression Examples (Phase 3+)**

Training data for compression:

```json
{
  "id": "compress_001",
  "messages": [
    {
      "role": "user",
      "content": "Summarize this conversation in 2 sentences:\n[full conversation text]"
    },
    {
      "role": "assistant",
      "content": "User asked about deployment strategies. I recommended blue-green deployment with automated rollback for their use case."
    }
  ],
  "metadata": {
    "category": "compression",
    "compression_ratio": 0.15
  }
}
```

### **Tool Use Examples (Phase 5+)**

If implementing tool calling:

```json
{
  "id": "tool_001",
  "messages": [
    {"role": "user", "content": "What's 15% of 2847?"},
    {
      "role": "assistant",
      "content": "<function_call>calculator(2847 * 0.15)</function_call>",
      "tool_calls": [{"name": "calculator", "args": {"expression": "2847 * 0.15"}}]
    },
    {"role": "tool", "content": "427.05"},
    {"role": "assistant", "content": "15% of 2847 is 427.05"}
  ],
  "metadata": {"category": "tool_use", "tools": ["calculator"]}
}
```

---

## Validation

### **Automated Checks**

**Script:** `scripts/validate_data.py`

**Checks:**
- [ ] Valid JSON
- [ ] Required fields present
- [ ] No duplicate IDs
- [ ] Token counts reasonable
- [ ] Category values valid
- [ ] Test set doesn't overlap train

**Run before training:**
```bash
python scripts/validate_data.py data/toy/train.jsonl
```

### **Manual Review**

**Spot-check:**
- 10-20 random examples
- Do they look right?
- Capture your actual patterns?
- Any obviously bad ones?

**Red flags:**
- Looks generic (could be anyone)
- Contradicts other examples
- Contains PII you don't want trained
- Wrong category

---

## Privacy and Safety

### **Do NOT Include**

**Never in training data:**
- ❌ API keys, passwords, secrets
- ❌ Customer/client PII (names, emails, data)
- ❌ Proprietary information
- ❌ Anything you wouldn't want public

**Be careful with:**
- ⚠️ Personal addresses, phone numbers
- ⚠️ Financial information
- ⚠️ Medical information
- ⚠️ Legal matters

### **Scrubbing**

If example is valuable but has sensitive info:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Can you email [REDACTED_EMAIL] about the project?"
    },
    {
      "role": "assistant",
      "content": "Sure, I'll send them an update on the current status."
    }
  ],
  "metadata": {"scrubbed": true, "scrubbed_fields": ["email"]}
}
```

---

## Conversion Scripts

### **From Slack Export**

```python
# scripts/convert_slack.py
import json

def slack_to_training(slack_export):
    """Convert Slack DM export to training format"""
    examples = []
    for msg in slack_export['messages']:
        if msg['user'] == 'you':
            # Extract user question and your response
            example = {
                "id": f"slack_{msg['ts']}",
                "messages": [
                    {"role": "user", "content": previous_msg['text']},
                    {"role": "assistant", "content": msg['text']}
                ],
                "metadata": {
                    "category": "style",
                    "source": "slack",
                    "timestamp": msg['ts']
                }
            }
            examples.append(example)
    return examples
```

### **From Discord**

```python
# scripts/convert_discord.py
# Similar to Slack, parse Discord export format
```

### **From Manual Entry**

```python
# scripts/create_example.py
# Interactive CLI to add examples manually
```

---

## Example: Complete Toy Dataset

**train.jsonl (first 3 lines):**
```jsonl
{"id":"toy_001","messages":[{"role":"user","content":"How do you approach debugging a production issue?"},{"role":"assistant","content":"First, don't panic. Check logs, reproduce if possible, understand scope (how many affected?). If critical, mitigate first (rollback/workaround), debug properly second. I value precision over speed under pressure."}],"metadata":{"category":"style","source":"manual","quality":"high"}}
{"id":"toy_002","messages":[{"role":"user","content":"What tools do you use daily?"},{"role":"assistant","content":"VSCode for coding, PowerShell for automation, Git for version control. Kaseya VSA and our custom HelixAPI for RMM work. Slack and Linear for communication and project tracking."}],"metadata":{"category":"facts","source":"manual","quality":"high"}}
{"id":"toy_003","messages":[{"role":"user","content":"Should we optimize this code now or ship and optimize later?"},{"role":"assistant","content":"Depends on the optimization. If it's a known bottleneck blocking users, optimize first. If it's premature (no evidence it's slow), ship and measure. Don't guess at performance, profile it."}],"metadata":{"category":"decisions","source":"manual","quality":"high"}}
```

**test.jsonl (first line):**
```jsonl
{"id":"test_001","messages":[{"role":"user","content":"What's your process for code reviews?"},{"role":"assistant","content":"Check for correctness first, then maintainability. I focus on logic errors, edge cases, and whether I can understand what it does 6 months from now. Nitpicks about style are low priority unless it's truly unreadable."}],"metadata":{"category":"style","source":"manual","quality":"high"}}
```

---

## Summary

**Key points:**
- JSONL format (one JSON per line)
- Conversation format (messages array)
- Rich metadata (category, source, quality)
- Four categories: style, facts, decisions, psychology
- Quality over quantity
- No PII or secrets

**For Phase 0a:**
- 50 train, 10 test
- Mixed categories (20/20/20 split)
- Real examples from actual conversations
- Format as above
- Save to `data/toy/`

**Validation before training:**
```bash
python scripts/validate_data.py data/toy/train.jsonl
python scripts/validate_data.py data/toy/test.jsonl
```

---

*This format will evolve as we learn. Update this doc if we discover better approaches.*
