# EAG Architecture

## Core Concept: Decoupling Planning and Execution

EAG restructures the agentic pipeline into two strictly separated phases:

```
┌─────────────────┐     ┌─────────────────┐
│   Planner       │     │   Executor      │
│   (LLM)         │────▶│   (Local)       │
│                 │     │                 │
│ • Schema-       │     │ • Read-only     │
│   constrained   │     │ • Stateless     │
│ • Intent-only   │     │ • No egress     │
│ • No raw data   │     │ • Pre-registered│
└─────────────────┘     │   operators     │
                        │                 │
                        │ ◀──── Results   │
                        │   (aggregated)  │
                        └─────────────────┘
```

## Zero-Data Boundary

### Key Invariants
1. **No Raw Data to LLM**: The planner receives only high-level operational intents, never raw datasets (patient histories, ledger entries, sensor telemetry).
2. **Schema-Constrained Output**: Plans are generated against a public, static contract enumerating permissible operations, data types, aggregation boundaries, and capability scopes.
3. **Local Execution**: A read-only execution environment parses, validates, and executes plans directly against the underlying information system.
4. **Output Isolation**: The execution boundary strips all raw payloads from the return path. Only pre-aggregated, structurally validated results are delivered to the planner.
5. **Deterministic Enforcement**: Security derives from structural constraints, not probabilistic safeguards or runtime filtering.

## Security Posture

EAG aligns with zero-trust principles by design:

| Traditional Agent | EAG |
|------------------|-----|
| Runtime isolation, output filtering | Structural constraints at plan level |
| Trust in model compliance | Trust never placed in model |
| Raw data flows to inference endpoint | Raw data never crosses trusted boundary |
| Vulnerable to adversarial prompting | Adversarial plans rejected at verification gate |

### Executor Constraints
- **Stateless processing engine** with no network egress
- **No persistent storage** or arbitrary system function invocation
- **Pre-registered operators** only; execution confined to validated aggregation pipelines
- **Schema-driven serialization** guarantees only aggregated, non-identifiable results are returned

## Threat Model

EAG assumes:
- The LLM planner may be compromised, drift, or receive adversarial inputs
- The local executor operates in a trusted, isolated environment
- The schema/contract is maintained by trusted domain operators

Under this model, even a compromised planner cannot:
- Exfiltrate raw data (never received)
- Execute arbitrary code (plans validated against schema)
- Bypass aggregation boundaries (enforced by executor)

---
*Next: [Validation & Security](./Validation-and-Security.md) for plan verification details.*
