---
description: How to hand off context between AI sessions
---

# Session Handoff Workflow

**When to use:** Starting a new AI session (Cascade, GPT plugin, etc.) or resuming work after a break.

---

## For the USER

### **Starting a New AI Session**

1. **Point the AI to the session context:**
   ```
   Before we start, read SESSION_CONTEXT.md - it has everything you need to know about the current state.
   ```

2. **Specify what you're working on:**
   ```
   We're currently on Phase 0 (environment setup). Follow docs/phases/phase0.md.
   ```

3. **If using GPT plugin for boilerplate:**
   ```
   Read SESSION_CONTEXT.md and docs/architecture.md, then implement [specific feature] exactly as designed.
   ```

### **Updating Session Context**

After significant work, update `SESSION_CONTEXT.md`:

```
Update SESSION_CONTEXT.md:
- We've completed Phase 0
- Phi-3-mini is downloaded and working
- First inference test successful
- Ready to start Phase 1
```

---

## For AI ASSISTANTS

### **On Session Start**

1. **Read SESSION_CONTEXT.md FIRST**
   - Contains current state, critical decisions, gotchas
   - More important than full docs for immediate context

2. **Check current phase:**
   - Look at "Where We Are" section
   - Read corresponding phase doc (e.g., `docs/phases/phase0.md`)

3. **Review "What NOT to Do"**
   - Avoid common pitfalls
   - Don't deviate from established architecture

### **Before Making Suggestions**

Check these docs in order:
1. `SESSION_CONTEXT.md` - Current state and quick decisions
2. `docs/decisions.md` - All major decisions with rationale
3. `docs/architecture.md` - Technical design details
4. `docs/phases/phaseN.md` - Current phase checklist

### **When Writing Code**

Follow standards in `SESSION_CONTEXT.md`:
- ✅ Phi-3 requires `trust_remote_code=True`
- ✅ No emojis (user rule)
- ✅ 4-bit quantization
- ✅ Log VRAM usage
- ✅ Include progress indicators

### **Before Suggesting Architecture Changes**

**DON'T:**
- Make up new approaches
- Suggest external APIs for Track A
- Over-engineer early phases
- Deviate from compression architecture

**DO:**
- Reference existing design docs
- Ask user if deviation needed
- Document any approved changes in decisions.md

---

## Quick Start for Common Scenarios

### **Scenario: "Let's start Phase 0"**

**Steps:**
1. Read `SESSION_CONTEXT.md` (confirm Phase 0 is next)
2. Open `docs/phases/phase0.md`
3. Walk through checklist with user
4. Follow code examples exactly
5. Update SESSION_CONTEXT.md when complete

### **Scenario: "Write the compression layer"**

**Steps:**
1. Read `SESSION_CONTEXT.md` (check current state)
2. Read `docs/architecture.md` section on compression
3. Review "Context Compression" in SESSION_CONTEXT.md
4. Implement as designed (don't invent new approach)
5. Test with examples from phase doc

### **Scenario: "Debug this error"**

**Steps:**
1. Check "Implementation Gotchas" in SESSION_CONTEXT.md
2. Verify Phi-3 has `trust_remote_code=True`
3. Check hardware constraints (VRAM limits)
4. Review similar examples in phase docs
5. Add new gotcha to SESSION_CONTEXT.md if novel

### **Scenario: "Should we use Mistral instead?"**

**Steps:**
1. Check `docs/decisions.md` - Decision 7 (Phi-3 chosen)
2. Rationale: Philosophical alignment, tests thesis
3. Answer: No, start with Phi-3 (Mistral is Track B, added later)
4. Don't second-guess documented decisions

---

## Documentation Hierarchy

**For immediate work:**
1. `SESSION_CONTEXT.md` ← Start here
2. `docs/phases/phaseN.md` ← Current phase
3. `docs/architecture.md` ← Technical details

**For understanding:**
1. `START_HERE.md` ← Project overview
2. `docs/hypothesis.md` ← What we're testing
3. `docs/conversation_summary.md` ← Historical context

**For reference:**
1. `docs/decisions.md` ← Why we chose X
2. `docs/model_comparison.md` ← Why Phi-3 vs Mistral
3. `docs/setup.md` ← Troubleshooting

---

## Maintaining Session Context

### **What to Update**

**Update SESSION_CONTEXT.md when:**
- ✅ Phase completed
- ✅ Major code written
- ✅ New gotcha discovered
- ✅ Architecture deviation (with rationale)
- ✅ Blocker encountered
- ✅ Hypothesis result learned

**Don't update for:**
- ❌ Minor bug fixes
- ❌ Documentation clarifications
- ❌ Questions/discussion (use conversation_summary.md)

### **How to Update**

```markdown
**2025-11-XX, [time]:**
- Phase 0 completed ✅
- Phi-3-mini downloaded and working
- Discovered gotcha: [description]
- Next: Start Phase 1 (data collection)
```

Keep "Current State" and "Current Blockers" accurate.

---

## Template for New Sessions

**User message:**
```
New session - read SESSION_CONTEXT.md first, then let's continue [task].
```

**AI should respond:**
```
[Summary of current state from SESSION_CONTEXT.md]
[Confirmation of task]
[Any questions before proceeding]
```

**Example:**
```
Read SESSION_CONTEXT.md. Confirms:
- Phase 0 complete, Phi-3-mini working
- Track A architecture (compression + local only)
- Ready to start Phase 1 (data collection)

You want to [task]. Proceeding with:
1. [step 1]
2. [step 2]
3. [step 3]

Questions before I start:
- [any clarifications needed]
```

---

## For Long Breaks (Weeks/Months)

If resuming after extended time:

1. **Read in order:**
   - `SESSION_CONTEXT.md` (where we are NOW)
   - `docs/conversation_summary.md` (how we got here)
   - `docs/hypothesis.md` (what we're testing)
   - Current phase doc

2. **Verify assumptions:**
   - Models still correct?
   - Hardware still available?
   - Dependencies still work?
   - Goals still valid?

3. **Update if needed:**
   - Technology may have changed
   - Better approaches may exist
   - Discuss with user before pivoting

---

## Anti-Patterns to Avoid

### **Don't Assume Context**
❌ "Let's continue where we left off"  
✅ "Read SESSION_CONTEXT.md - we're on Phase 0, environment setup"

### **Don't Reinvent**
❌ "I think we should use a different compression approach"  
✅ "Architecture is designed in docs/architecture.md - implementing as specified"

### **Don't Over-Engineer**
❌ "Let's add Docker and CI/CD now"  
✅ "SESSION_CONTEXT.md says defer MLOps to Phase 4+"

### **Don't Second-Guess**
❌ "Mistral might be better than Phi-3"  
✅ "docs/decisions.md explains why Phi-3 was chosen (Decision 7)"

---

## Success Criteria

**Good session handoff:**
- ✅ AI read SESSION_CONTEXT.md before responding
- ✅ AI confirmed current state accurately
- ✅ AI followed existing architecture
- ✅ No need to re-explain decisions
- ✅ Work continued seamlessly

**Poor session handoff:**
- ❌ AI suggested already-decided alternatives
- ❌ AI didn't know current phase
- ❌ AI reinvented architecture
- ❌ User had to repeat context
- ❌ Wasted time re-explaining

---

**Use this workflow every time you start a new session or switch AI assistants.**

**Goal: Zero context loss between sessions.**
