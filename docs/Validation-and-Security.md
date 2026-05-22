# Validation & Security

## Dual-Stage Validation Pipeline

EAG guarantees executable correctness while minimizing hallucination through a two-phase validation process:

### Stage 1: Grammar-Constrained Decoding (GCD)
- **Input**: Static schema definitions and capability registries describing available operations, data types, and aggregation boundaries.
- **Mechanism**: Grammar-constrained decoding restricts the model's output space to syntactically valid plan templates aligned with structural contracts.
- **Result**: Format deviations eliminated at the token level; only schema-compliant plans can be generated.

### Stage 2: Static Plan Verifier
- **Input**: Syntactically valid plan from Stage 1.
- **Checks**:
  - Unreachable references
  - Invalid aggregation chains
  - Resource-exceeding patterns
  - Semantic alignment against shared capability registry
- **Result**: Proactive enforcement replaces reactive error-handling. Malformed or adversarial plans are rejected before execution.

```
Planner Output
      │
      ▼
┌─────────────────┐
│ GCD Decoder     │
│ (Syntax Check)  │
└────────┬────────┘
         │ Valid
         ▼
┌─────────────────┐
│ Static Verifier │
│ (Semantic Check)│
└────────┬────────┘
         │ Pass
         ▼
    Executor
```

## Executor Constraints

The read-only execution environment enforces strict boundaries:

| Constraint | Enforcement |
|------------|-------------|
| **No network egress** | Runtime isolation; no outbound connections permitted |
| **No persistent storage** | Ephemeral execution context; results aggregated in-memory |
| **No arbitrary functions** | Only pre-registered operators in capability registry |
| **Resource bounds** | Time/memory quotas enforced at operator level |
| **Output sanitization** | Schema-driven serialization; only aggregated, non-identifiable results returned |

## Deterministic Security Enforcement

Unlike probabilistic safeguards (prompt engineering, output filtering), EAG's security derives from structural invariants:

1. **Zero-Data by Construction**: Raw data never enters the planner's context window.
2. **Plan Verification Gate**: Invalid plans cannot reach execution.
3. **Executor Isolation**: Even if a malicious plan bypasses verification, the executor cannot exfiltrate data.
4. **Output Aggregation**: Results are pre-processed to remove identifiable information before returning to the planner.

### Adversarial Resilience
- **Malformed intents**: Rejected at verification stage with structured error signatures.
- **Hallucinated plans**: GCD prevents syntactically invalid outputs; verifier catches semantic violations.
- **Resource exhaustion**: Quotas enforced at operator level, independent of inference runtime.
- **Schema drift**: Capability registry versioning ensures planner and executor remain aligned.

## Verification Layer Independence

The verification layer operates independently of the inference runtime:
- No shared state with the LLM planner
- Cannot be bypassed by compromised or drifting models
- Structured error feedback enables iterative refinement without raw data exposure

---
*Next: [Feedback & Debugging](./Feedback-and-Debugging.md) for operator diagnostic workflows.*
