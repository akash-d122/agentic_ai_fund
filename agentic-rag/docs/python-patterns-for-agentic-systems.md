# Python Patterns for Agentic Systems

### Async Tasks
- retrieval
- API calls
- validation
- streaming

### Benefits
- lower latency
- better throughput
- parallel workflows

---

## 4. Structured Outputs

### Problem
Raw LLM text is unreliable.

### Solution
Force:
- JSON
- schemas
- typed responses

### Benefits
- easier orchestration
- deterministic parsing
- safer automation

---

## 5. State Management

### State Includes
- conversation history
- tool outputs
- memory
- execution steps

### Failure Mode
Stateless systems lose reasoning continuity.

---

## 6. Observability

### Log:
- prompts
- latency
- retrieval scores
- token usage
- tool failures
- hallucinations

### Why
AI debugging without observability is impossible.

---

## 7. Queue-Based Architecture

### Why
Long-running workflows block systems.

### Pattern
API -> Queue -> Worker -> Result Store

### Benefits
- resilience
- scaling
- retry safety
- async execution