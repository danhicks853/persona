# Phase 0a Training Data

**Goal:** 60 examples total (15 style, 15 facts, 30 decisions/reasoning)

**CRITICAL:** Using Qwen3-1.7B which has "thinking mode" - we need 50/50 reasoning/non-reasoning mix!

---

## Files

- **`template_examples.jsonl`** - Copy this and fill in your real examples
- **`train.jsonl`** - Final training set (50 examples) - create this when done
- **`test.jsonl`** - Final test set (10 examples) - create this when done

---

## Quick Process

1. **Edit template_examples.jsonl**
   - Replace the template text with your actual examples
   - Follow the NON-REASONING format for style/facts (direct answers)
   - Follow the REASONING format for decisions (use `<thinking>` tags!)
   - Add more lines following the same format
   - Keep going until you have 60 examples (15+15+30)

2. **When you have 60 examples:**
   - First 50 → `train.jsonl`
   - Last 10 → `test.jsonl`

3. **Verify format:**
   ```bash
   python -c "import json; [json.loads(line) for line in open('train.jsonl')]"
   ```

---

## Format Reminder

### NON-REASONING (style/facts - 30 examples):
```json
{"id":"style_001","messages":[{"role":"user","content":"question"},{"role":"assistant","content":"direct response"}],"metadata":{"category":"style","source":"slack","context":"work","quality":"high","reasoning":"false"}}
```

### REASONING (decisions - 30 examples):
```json
{"id":"decision_001","messages":[{"role":"user","content":"question"},{"role":"assistant","content":"<thinking>\nAnalysis here\n</thinking>\n\nFinal answer"}],"metadata":{"category":"decisions","source":"manual","context":"work","quality":"high","reasoning":"true"}}
```

**Important:**
- ONE line per example (no line breaks in the middle of JSON!)
- Use `\"` for quotes inside strings
- Use `\n` for line breaks inside the thinking tags
- Each ID must be unique

---

## Categories

### Style (15 examples) - NON-REASONING
How you write, communicate, explain things - DIRECT responses

### Facts (15 examples) - NON-REASONING
Information about you, your background, experience - FACTUAL answers

### Decisions/Psychology (30 examples) - REASONING
How you think through problems - SHOW YOUR THINKING PROCESS with `<thinking>` tags!

---

See **`../../docs/DATA_COLLECTION_GUIDE.md`** for full instructions!
