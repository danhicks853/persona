# Data Collection Guide - Phase 0a

**Goal:** Collect 60 real examples of how YOU write, think, and decide.

**Time estimate:** 2-3 hours

**Format:** See `data_format.md` for technical spec - this guide is the practical "how-to"

**CRITICAL:** We're using Qwen3-1.7B which has "thinking mode" for reasoning. We need a 50/50 mix of reasoning vs non-reasoning examples to preserve this capability while teaching it your style/facts.

---

## Quick Start

### **What You Need**

1. Access to your actual communications (Slack, Discord, email, etc.)
2. A text editor (Notepad, VS Code, whatever)
3. 1-2 hours of time

### **The Process**

1. Create the data directory
2. Collect raw examples
3. Format as JSONL
4. Split into train/test
5. Done!

---

## Step 1: Create Data Directory

```powershell
# From the persona directory
New-Item -ItemType Directory -Path "data/phase0a" -Force
```

---

## Step 2: Collect 60 Examples

**NEW RATIO:** 15 style + 15 facts + 30 decisions (50% reasoning, 50% non-reasoning)

### **Category A: Style (15 examples) - NON-REASONING**

**What:** How you write and communicate (direct responses, no thinking process)

**Where to look:**
- Slack DMs or channel messages
- Discord conversations
- Email responses
- GitHub PR comments
- Quick help/answers you've given

**What to capture:**
- Your actual writing (not formal documents)
- Direct answers, explanations
- Various tones (helping, technical, casual)
- NO chain-of-thought - just natural responses

**Template:**
```
USER: Can you help me understand this error message?
YOU: Sure! That error means the script can't find the file path you specified. 
Check line 15 - you probably need to use the full path instead of a relative one.

Category: Style
Context: Work, technical help
Type: NON-REASONING (direct answer)
```

### **Category B: Facts (15 examples) - NON-REASONING**

**What:** Information about you (direct factual answers)

**Examples:**
- "What do you do?" → Direct answer
- "What are your skills?" → List them
- "What projects?" → Name them
- NO analysis, just facts

**Template:**
```
USER: What do you do for work?
YOU: I'm an automation engineer at an MSP. I work with PowerShell, RMM 
platforms like Kaseya and Datto, and build internal tools for the team.

Category: Facts
Context: Work background
Type: NON-REASONING (factual)
```

### **Category C: Decisions/Psychology (30 examples) - REASONING** ⭐

**What:** How you think through problems (SHOW YOUR REASONING PROCESS)

**CRITICAL:** Use `<thinking>` tags to show your mental process!

**Examples:**
- "Should we migrate now or wait?" → Show analysis
- "Which tool?" → Show trade-off evaluation
- "How to prioritize?" → Show decision framework
- Real choices you've made with your reasoning exposed

**Template with THINKING:**
```
USER: We have a tight deadline. Should we automate this process or do it manually?
YOU: <thinking>
Let me analyze this:

Factors:
- Deadline: 2 days
- Automation time: 8 hours build + 2 hours test = 10 hours
- Manual time: 4 hours total
- Future use: Will we need this again? Yes

My heuristics:
- Accuracy > Speed (but deadline is real)
- Don't let perfect be the enemy of done
- Technical debt is okay if documented

Trade-offs:
- Automate now: Risk missing deadline, but solved forever
- Manual now: Meets deadline, but repeating work later

Decision weight: Deadline is the constraint. After deadline, automate it.
</thinking>

Manual for now. Here's why:

The deadline in 2 days is a hard constraint. Manual takes 4 hours, automation 
takes 10 hours. We can't risk missing the deadline.

BUT - we document this as technical debt and schedule automation for right 
after the deadline. That way we never do it manually again. Don't let perfect 
be the enemy of done, but also don't forget to fix it after.

Category: Decisions
Context: Work, time pressure, prioritization
Type: REASONING (chain-of-thought shown)
```

**Keys for reasoning examples:**
- Show your thinking in `<thinking>` tags
- Expose your mental models
- Show trade-off analysis
- Reveal your heuristics/values
- Demonstrate how you actually think

---

## Step 3: Format as JSONL

### **Manual Method (Simple)**

1. Open a text editor
2. Create `data/phase0a/raw_examples.jsonl`
3. For each example, add a line like this:

```json
{"id":"style_001","messages":[{"role":"user","content":"Can you help me understand this error message?"},{"role":"assistant","content":"Sure! That error means the script can't find the file path you specified. Check line 15 - you probably need to use the full path instead of a relative one."}],"metadata":{"category":"style","source":"slack","context":"work"}}
```

**Each line is ONE complete JSON object - no line breaks within an object!**

### **Easy Template**

Copy this and fill in your examples:

```json
{"id":"style_001","messages":[{"role":"user","content":"[QUESTION]"},{"role":"assistant","content":"[YOUR RESPONSE]"}],"metadata":{"category":"style","source":"slack","context":"work"}}
{"id":"style_002","messages":[{"role":"user","content":"[QUESTION]"},{"role":"assistant","content":"[YOUR RESPONSE]"}],"metadata":{"category":"style","source":"discord","context":"casual"}}
{"id":"facts_001","messages":[{"role":"user","content":"[QUESTION]"},{"role":"assistant","content":"[YOUR RESPONSE]"}],"metadata":{"category":"facts","source":"manual","context":"personal"}}
{"id":"decision_001","messages":[{"role":"user","content":"[SITUATION/QUESTION]"},{"role":"assistant","content":"[YOUR DECISION + REASONING]"}],"metadata":{"category":"decisions","source":"manual","context":"work"}}
```

**Tips:**
- Use `\"` for quotes inside strings: `"I think \"automation\" is key"`
- Keep it on ONE line per example
- Use simple IDs: `style_001`, `style_002`, `facts_001`, etc.

---

## Step 4: Split Train/Test

### **Option A: Manual Split**

1. Copy your `raw_examples.jsonl`
2. Take the first 50 lines → save as `data/phase0a/train.jsonl`
3. Take the last 10 lines → save as `data/phase0a/test.jsonl`

### **Option B: Random Split (I can write a script)**

If you want, I can create a Python script that:
- Reads your raw examples
- Randomly shuffles them
- Splits 50 train / 10 test
- Saves to separate files

Let me know if you want this!

---

## Example: Complete Entry

Here's a real example properly formatted:

```json
{"id":"style_001","messages":[{"role":"user","content":"The PowerShell script keeps failing on line 23. Any ideas?"},{"role":"assistant","content":"Check if the variable is actually populated before you use it. Add this before line 23: if (-not $variableName) { Write-Error 'Variable is null'; return }. That'll tell you if it's a null reference issue."}],"metadata":{"category":"style","source":"slack","context":"work","quality":"high"}}
```

Broken down:
- **id**: `style_001` (unique identifier)
- **messages**: Array with user question and your response
- **metadata**: 
  - category: `style` (or `facts` or `decisions`)
  - source: where it came from
  - context: work/personal/technical/casual
  - quality: high/medium/low (your judgment)

---

## Quality Guidelines

### **Good Examples:**
- ✅ Real communication you've actually written
- ✅ Reflects how you naturally express yourself
- ✅ Shows your reasoning or knowledge
- ✅ Diverse contexts and topics
- ✅ Honest representation

### **Avoid:**
- ❌ Formal documents or official writing
- ❌ Generic responses anyone could give
- ❌ Sanitized or "professional" voice if that's not natural
- ❌ Extremely short responses (unless that's authentic)
- ❌ Duplicates or near-duplicates

---

## Quick Tips

1. **Don't overthink it** - If you wrote it and it sounds like you, it's good
2. **Be authentic** - Include your mannerisms, phrases, style quirks
3. **Variety matters** - Different topics, tones, lengths
4. **Real is better than perfect** - Actual communication > crafted examples
5. **Privacy** - Remove any sensitive info (names, passwords, proprietary stuff)

---

## Validation Checklist

Before you're done, check:

- [ ] 60 total examples (15 style, 15 facts, 30 decisions/reasoning)
- [ ] 30 reasoning examples have `<thinking>` tags showing your thought process
- [ ] 30 non-reasoning examples are direct responses
- [ ] Each is valid JSONL (one JSON object per line)
- [ ] IDs are unique (`style_001`, `style_002`, etc.)
- [ ] Split into train.jsonl (50) and test.jsonl (10)
- [ ] No sensitive information included
- [ ] Examples are actually representative of you
- [ ] Reasoning examples show HOW you think, not just WHAT you decide

---

## What's Next?

After data collection:

1. **Validate format** - I can check if your JSONL is valid
2. **Review examples** - Spot check a few to ensure quality
3. **Start training** - Run the training script with your data!

---

## Need Help?

**Common issues:**

**Q: "My quotes are breaking the JSON"**
A: Use `\"` instead of `"` inside strings

**Q: "The file won't load"**
A: Each example must be on ONE line (no line breaks in the middle)

**Q: "I don't have 60 examples"**
A: Start with what you have! Even 30-40 can work for the toy project

**Q: "Can I collect more later?"**
A: Absolutely! Phase 0a is just to test the pipeline

---

**You got this! Real examples of how YOU communicate and think. That's all we need.**
