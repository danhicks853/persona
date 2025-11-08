# Windsurf Workflows

Workflows help maintain consistency and process across AI sessions.

## Available Workflows

### **session-handoff.md** (`/session-handoff`)
How to hand off context between AI sessions (Cascade → GPT plugin, or resuming after break).

**When to use:**
- Starting new AI session
- Switching AI assistants
- Resuming after days/weeks
- GPT plugin needs context

**Key file:** `SESSION_CONTEXT.md` (root of repo)

---

## How to Use Workflows

**In chat:**
```
/session-handoff
```

Or manually:
```
Read .windsurf/workflows/session-handoff.md
```

---

## Workflow Philosophy

**Workflows document:**
- ✅ Repeatable processes
- ✅ Common scenarios
- ✅ Decision trees
- ✅ Anti-patterns to avoid

**Workflows don't:**
- ❌ Replace docs (they reference docs)
- ❌ Contain architecture (see docs/architecture.md)
- ❌ Make decisions (see docs/decisions.md)

**Think of workflows as "how to use the docs effectively."**

---

## Future Workflows (To Be Created)

- `phase-checklist.md` - How to complete each phase systematically
- `debugging.md` - Common issues and solutions
- `data-pipeline.md` - How to run the data collection/curation pipeline
- `evaluation.md` - How to run the comparison evaluation

These will be created as needed during development.

---

*Workflows are living documents - update as processes improve.*
