# Architecture Design

**Last Updated:** 2025-11-08 (Morning session)

---

## Overview

This document describes the technical architecture for both experimental approaches, with emphasis on the neurosymbolic design and context compression system.

---

## Model Selection

### **Chosen Models**

**Track A (Neurosymbolic):** Microsoft Phi-3-mini-4k-instruct
- **Size:** 3.8B parameters
- **Context:** 4K tokens (~3,000 words)
- **Philosophy:** Quality data > quantity (trained on curated textbooks)
- **Rationale:** Aligns with project thesis (structure > scale), fast iteration, efficient

**Track B (Traditional Baseline):** Mistral-7B-Instruct-v0.2
- **Size:** 7B parameters
- **Context:** 32K tokens (~24,000 words)
- **Philosophy:** Efficient attention, large context window
- **Rationale:** Proven fine-tuning performance, community support, not gated

### **Why These Choices?**

**Phi-3-mini advantages:**
- ✅ Philosophical alignment (quality > scale)
- ✅ Faster downloads, training, inference
- ✅ Proves efficiency thesis if it works
- ✅ Lower VRAM usage = more experimentation headroom
- ✅ Microsoft research validates small-model viability

**Mistral-7B advantages:**
- ✅ Larger capacity for complex reasoning
- ✅ 32K context = less pressure on compression system
- ✅ Well-documented fine-tuning procedures
- ✅ Not gated (immediate download)
- ✅ Strong community support

**Comparison approach:**
- Build both in parallel (A/B testing)
- Use identical test sets
- Compare quality, efficiency, explainability
- Determine if small+structure matches medium+scale

---

## Context Compression Architecture

### **The Problem**

**Phi-3 has 4K token limit:**
- Average conversation turn: 200-500 tokens
- 4K = ~10-15 turns before limit
- For complex tasks (debugging scripts, design discussions), hits limit fast

**Traditional solution:** Use bigger model with 32K+ context (Mistral, Claude)

**Our solution:** Context compression + external memory

---

### **Inspiration: Human Memory**

**How humans handle limited working memory:**

```
Working Memory (active context):
- Last 5-10 conversational turns
- Current task details
- Immediately relevant facts
(~4K tokens equivalent)

Long-Term Memory (external storage):
- Summarized/compressed old conversations
- Structured knowledge (facts, patterns, decisions)
- Full transcripts archived
(unlimited storage)

Retrieval Process:
- When needed, fetch from long-term memory
- Reconstruct details on demand
- Compress old working memory to long-term storage
```

**This is exactly what we'll implement for the model.**

---

### **Architecture: Three-Tier Memory System**

```
┌─────────────────────────────────────────────────────────┐
│ Active Context (Phi-3's 4K window)                      │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Psychology Profile      [~500 tokens, always loaded]│ │
│ │ Recent Conversation     [~1,500 tokens, last 5 turns]│ │
│ │ Retrieved Memories      [~1,000 tokens, relevant]   │ │
│ │ Current Task/Query      [~500 tokens]               │ │
│ │ Tool Results (if any)   [~500 tokens]               │ │
│ └─────────────────────────────────────────────────────┘ │
│ Total: ~3,500-4,000 tokens (fits in 4K limit)          │
└─────────────────────────────────────────────────────────┘
                          ↓ compress when full
┌─────────────────────────────────────────────────────────┐
│ Compressed Memory (summaries)                           │
│ - Conversation summaries (key points, decisions)        │
│ - Pattern extraction (repeated behaviors)               │
│ - Decision logs (what was chosen, why)                  │
│ - Fast retrieval via embeddings                         │
└─────────────────────────────────────────────────────────┘
                          ↓ archive raw
┌─────────────────────────────────────────────────────────┐
│ Raw Archive (full fidelity)                             │
│ - Complete conversation history (never deleted)         │
│ - Full context for each decision                        │
│ - Versioned with timestamps                             │
│ - Retrievable when detail needed                        │
└─────────────────────────────────────────────────────────┘
```

---

### **How It Works (Detailed Flow)**

#### **Turn 1-5: Normal Operation**
```python
# Active context stays under 4K
for turn in range(1, 6):
    user_input = get_user_message()
    context = build_context(
        psychology=load_profile(),
        recent=last_5_turns,
        current=user_input
    )
    response = model.generate(context)
    store_turn(turn, user_input, response)
```

#### **Turn 6-10: Approaching Limit**
```python
# Context grows, approaching 3,500 tokens
if estimate_tokens(active_context) > 3500:
    trigger_compression()
```

#### **Compression Process**
```python
def compress_context():
    # 1. Identify old turns (keep last 5 fresh)
    old_turns = conversation_history[:-5]
    
    # 2. Compress using model itself
    summary = model.generate(f"""
    Summarize this conversation preserving:
    - Key decisions and rationale
    - Important facts learned
    - Emotional context
    - Open questions
    Keep under 500 tokens.
    
    Conversation:
    {old_turns}
    """)
    
    # 3. Store THREE versions
    compressed_store.add(summary)        # Fast retrieval
    raw_archive.store(old_turns)         # Full fidelity
    embeddings.add(summary)              # Semantic search
    
    # 4. Update active context
    active_context = [
        {"system": f"Previous context: {summary}"}
    ] + conversation_history[-5:]  # Last 5 turns verbatim
```

#### **Retrieval When Needed**
```python
def retrieve_memory(query):
    # Semantic search in compressed memory
    relevant = embeddings.search(query, top_k=3)
    
    # If user needs details, fetch from raw archive
    if needs_detail(query):
        raw = raw_archive.get(relevant.ids)
        return raw
    else:
        return relevant.summaries
```

---

### **Key Design Decisions**

#### **1. Why Store Raw + Compressed?**

**Compression is lossy:**
- Summarization drops details
- Edge cases get lost
- Errors can propagate

**But compression is necessary:**
- Can't fit everything in 4K context
- Need fast inference (less tokens = faster)

**Solution:** Store both
- Compressed for speed (in active context)
- Raw for accuracy (retrievable when needed)

#### **2. Why Keep Last 5 Turns Verbatim?**

**Recent context is most important:**
- Current task depends on recent discussion
- Subtle details matter
- User expects continuity

**Older context can be summarized:**
- High-level gist is sufficient
- Details retrievable if needed

#### **3. Why Use Model For Compression?**

**Advantages:**
- Understands semantic importance
- Can extract key decisions/facts
- Maintains conversational flow

**Risk:** Model might compress poorly, lose critical info

**Mitigation:**
- Test compression quality (unit tests)
- Compare compressed vs raw periodically
- Use deterministic extractive summarization as fallback

---

## Tool Use Architecture (Track B Only)

**Track A (Phase 0-4):** NO external tools, pure local

**Track B (Phase 5+, optional):** Add tools with guardrails

### **Tool Layer Design**

```python
class ToolManager:
    def __init__(self, mode="offline"):
        self.mode = mode  # "offline" or "hybrid"
        self.tools = {
            "search": GoogleSearch() if mode == "hybrid" else None,
            "deep_reasoning": ClaudeAPI() if mode == "hybrid" else None,
            "code_exec": LocalExecutor(),  # Always available
            "memory": KnowledgeGraph()     # Always available
        }
        self.cost_cap = 10.00  # Monthly cap
        self.user_consent_required = True
    
    def call_tool(self, tool_name, params):
        # Check mode
        if self.mode == "offline" and tool_name in ["search", "deep_reasoning"]:
            return "Tool unavailable in offline mode"
        
        # Check cost
        if self.tools[tool_name].cost > 0:
            if self.total_cost_this_month >= self.cost_cap:
                return "Monthly cost cap reached"
        
        # Request consent
        if self.user_consent_required:
            if not ask_user_permission(tool_name, params):
                return "User declined tool use"
        
        # Execute
        result = self.tools[tool_name].execute(params)
        
        # Log for transparency
        self.log_tool_call(tool_name, params, result, cost)
        
        return result
```

### **Offline Fallback**

**If Track A (offline) is sufficient, Track B may never be needed.**

**Degraded but functional:**
- Can't call Claude for deep reasoning → use Phi-3 (less capable but works)
- Can't Google search → rely on stored knowledge only
- Can't access external APIs → local alternatives only

---

## Knowledge Graph Design

**Purpose:** Store structured facts, decisions, patterns

### **Schema (Simplified)**

```python
class KnowledgeGraph:
    nodes = {
        "facts": [
            {"id": "fact_001", "content": "VSA migration planned for Q2 2026", "source": "meeting_2024-10-15", "confidence": 0.9}
        ],
        "decisions": [
            {"id": "decision_001", "situation": "script failed on 20 endpoints", "chosen": "rollback immediately", "rationale": "stability > speed", "date": "2024-11-01"}
        ],
        "patterns": [
            {"id": "pattern_001", "trigger": "time pressure", "response": "ship hot-fix, refactor later", "frequency": 0.7}
        ],
        "relationships": [
            {"person": "Dan", "works_at": "MSP", "role": "Automation Engineer"}
        ]
    }
    
    def query(self, question):
        # Graph traversal + semantic search
        return relevant_nodes
```

### **Integration With Compression**

```python
def compress_and_extract(conversation):
    # 1. Create summary
    summary = summarize(conversation)
    
    # 2. Extract structured knowledge
    extracted = extract_entities(conversation, schema={
        "decisions": ["situation", "chosen", "rejected", "rationale"],
        "facts": ["claim", "source", "confidence"],
        "patterns": ["trigger", "response"]
    })
    
    # 3. Store in graph
    for decision in extracted["decisions"]:
        knowledge_graph.add_decision(decision)
    
    # 4. Store summary in vector DB
    vector_store.add(summary)
    
    # 5. Archive raw
    raw_archive.store(conversation)
```

---

## Psychology Framework Integration

**The psychology profile is ALWAYS in active context (500 tokens reserved).**

### **Profile Structure**

```json
{
  "traits": {
    "big_five": {"openness": 0.78, "conscientiousness": 0.91, ...},
    "attachment": "anxious-avoidant",
    "cognitive_biases": ["confirmation_bias", "sunk_cost_fallacy"]
  },
  "values": {
    "hierarchy": ["accuracy", "maintainability", "speed", "user_experience"],
    "weights": [0.4, 0.25, 0.25, 0.1]
  },
  "trauma_patterns": {
    "triggers": ["sudden escalation", "lack of control"],
    "responses": ["hyper-documentation", "defensive communication"],
    "coping": ["detailed planning", "multiple contingencies"]
  },
  "heuristics": {
    "under_stress": "default_to_precision_over_speed",
    "time_constrained": "ship_then_refactor",
    "interpersonal_conflict": "document_then_deescalate"
  }
}
```

### **How Model Uses It**

```python
def generate_response(context, psychology_profile):
    # Profile influences:
    # 1. Tone (conscientiousness → thorough, precise)
    # 2. Risk assessment (trauma patterns → cautious in certain contexts)
    # 3. Decision weights (values hierarchy → optimize for accuracy first)
    # 4. Behavioral responses (heuristics → if stressed, be more careful)
    
    prompt = f"""
    You are Dan. Your personality:
    {psychology_profile}
    
    Current situation:
    {context}
    
    Respond as Dan would, considering his traits, trauma patterns, and heuristics.
    """
    
    return model.generate(prompt)
```

---

## Evaluation Architecture

### **Automated Metrics**

```python
class Evaluator:
    def style_similarity(self, model_output, real_dan_examples):
        # Sentence length distribution
        # Word choice patterns
        # Formality score
        # Emoji usage (none per user rules)
        return similarity_score
    
    def decision_accuracy(self, model_decisions, dan_would_do):
        # 30 scenario battery
        # Binary: match or not
        return accuracy_percentage
    
    def hallucination_rate(self, model_claims, knowledge_graph):
        # Check if claims are backed by stored facts
        # Or explicitly states uncertainty
        return hallucination_percentage
```

### **Human Evaluation**

```python
# Blind A/B testing
def turing_test():
    for prompt in test_set:
        response_a = phi_model.generate(prompt)  # Neurosymbolic
        response_b = mistral_model.generate(prompt)  # Traditional
        response_c = real_dan_response(prompt)  # Ground truth
        
        # Shuffle and present to evaluators
        responses = shuffle([response_a, response_b, response_c])
        
        for evaluator in [friend1, friend2, friend3]:
            evaluator.rate(responses)  # Which is most "Dan-like"?
    
    # Calculate pass rate
    return accuracy
```

---

## Publishing Strategy

### **Three-Tier Release**

**Tier 1: Base Model (HuggingFace)**
```
huggingface.co/dan-hicks/persona-phi3
```
- Fine-tuned Phi-3 weights (LoRA adapters only)
- Standard format, works anywhere
- Maximum compatibility
- No external dependencies

**Tier 2: Full System (GitHub + PyPI)**
```
pip install persona-ai
```
- Complete architecture (compression + memory + psychology)
- Python library
- Full features for developers
- Documented API

**Tier 3: Reference API (Docker)**
```
docker run danhicks/persona-api
```
- OpenAI-compatible REST API
- Self-hostable
- Language agnostic
- Demonstrates full system

### **Compatibility Matrix**

| Use Case | Method | Compatibility |
|----------|--------|---------------|
| Just want Dan's voice | HF model | ✅ Universal |
| Want compression system | Python lib | ✅ Python only |
| Want full neurosymbolic | Python lib | ✅ Python only |
| Want API access | Docker/hosted | ✅ Any language |
| Integrate with LangChain | Python lib | ✅ Custom wrapper |
| Use in Ollama/LMStudio | HF model | ✅ Convert to GGUF |

---

## Data Pipeline

### **Semi-Automated Curation**

```python
# Phase 1: Export
raw_data = {
    "slack": export_slack_history(),        # 10K+ messages
    "tickets": export_freshdesk_tickets(),  # 1K+ tickets
    "github": export_pr_comments(),         # 500+ comments
    "notes": export_personal_notes()        # Varied
}

# Phase 2: Weak Labeling (Automated)
candidates = {
    "style_examples": detect_style_patterns(raw_data),  # 2K candidates
    "decisions": detect_decision_pairs(raw_data),        # 1K candidates
    "psych_examples": detect_emotional_patterns(raw_data)  # 500 candidates
}

# Phase 3: Manual Review (Spot Check)
validated = {
    "style": human_review(candidates["style_examples"][:200]),  # Review 10%
    "decisions": human_review(candidates["decisions"][:100]),   # Review 10%
    "psych": human_review(candidates["psych_examples"][:50])    # Review 10%
}

# Phase 4: Confidence Filtering
high_confidence_weak = filter_by_confidence(candidates, threshold=0.85)

# Phase 5: Training Set
training_data = merge(validated, high_confidence_weak)
```

**Result:** 1-2K style examples, 500+ decision pairs, 200+ psych examples with manageable manual effort

---

## MLOps (Deferred to Phase 4+)

**Phase 0-3:** Keep it simple
- Git for version control
- Config files for hyperparameters
- Notebook for experiment tracking

**Phase 4+** (if publishing):
- Docker for reproducibility
- Weights & Biases for experiment tracking
- DVC for data versioning
- CI/CD for tests

**Don't over-engineer early phases.**

---

## Next Steps

1. ✅ Architecture documented
2. → Implement Phase 0 (get Phi-3 running)
3. → Build compression layer (Phase 3)
4. → Add knowledge graph (Phase 3-4)
5. → Integrate psychology (Phase 3-4)
6. → Compare with Mistral (Phase 5)

---

*Architecture v1.0 - 2025-11-08*
*Incorporates: Model selection, context compression, Track A/B split, publishing strategy*
