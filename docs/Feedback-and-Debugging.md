# Feedback & Debugging

## Metadata-Bound Feedback Mechanisms

Operational deployments require continuous alignment between planner capabilities and evolving infrastructure states. Traditional debugging workflows inherently expose raw telemetry, violating EAG's zero-data boundary. EAG introduces metadata-bound feedback that enables diagnosis and correction without data isolation violations.

### Structured Diagnostics vs. Raw Logs

When a plan fails verification or yields inconclusive results, the executor emits **structured diagnostics** instead of raw log excerpts:

| Diagnostic Type | Content | Purpose |
|----------------|---------|---------|
| **Capability Mismatch** | Missing operator, invalid parameter type, unsupported aggregation | Guide planner to use available capabilities |
| **Schema Violation** | Invalid field reference, type mismatch, boundary exceedance | Enforce contract compliance |
| **Aggregation Error** | Empty result set, cardinality violation, null propagation | Improve plan robustness |
| **Resource Exhaustion** | Timeout, memory limit, operator quota exceeded | Enable adaptive resource planning |

### Operator Feedback Loop

Domain operators can intervene at the abstraction layer:

1. **Annotate Diagnostics**: Add constraint hints or correction suggestions to structured errors.
2. **Inject Constraints**: Adjust public capability contract (e.g., add new operator, modify aggregation boundary).
3. **Iterative Refinement**: Planner consumes updated contract in subsequent iterations without raw data exposure.

```
Plan Failure
      │
      ▼
┌─────────────────┐
│ Executor emits  │
│ structured diag │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Operator annotates │
│ or adjusts contract│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Planner refines │
│ with updated spec│
└────────┬────────┘
         │
         ▼
    Retry Execution
```

## Decoupling Transparency from Visibility

EAG preserves operational transparency while maintaining data isolation:

- **No Raw Telemetry**: Operators never see patient records, financial transactions, or sensor streams.
- **Abstracted Observability**: System health, plan success rates, and verification metrics are aggregated and anonymized.
- **Human-in-the-Loop**: Operators guide refinement via constraint hints, not data inspection.

### Example: Clinical Workflow Debugging

**Scenario**: Plan fails to aggregate patient vitals due to schema mismatch.

**Traditional Approach**: 
- Expose raw patient data to operator for manual inspection
- Operator identifies field name discrepancy
- Risk: HIPAA violation, data exfiltration surface

**EAG Approach**:
- Executor emits: `{"error": "schema_violation", "field": "vitals.heart_rate", "expected_type": "numeric[]", "received": "string"}`
- Operator annotates: `{"hint": "use aggregation: mean(vitals.heart_rate)"}`
- Planner refines plan using hint; no raw data exposed

## Iterative Plan Refinement

The feedback loop enables continuous improvement:

1. **Initial Plan**: Planner generates schema-compliant plan based on current capability registry.
2. **Execution Attempt**: Executor validates and runs plan; returns aggregated results or structured error.
3. **Feedback Integration**: Operator annotations or contract updates are versioned and propagated.
4. **Refined Plan**: Planner incorporates feedback; generates improved plan without raw data exposure.

### Versioning and Auditability

- **Capability Registry Versioning**: Each plan execution references a specific contract version.
- **Diagnostic Provenance**: Structured errors include schema version, operator ID, and timestamp.
- **Refinement History**: Plan iterations are logged with metadata-only traces for compliance auditing.

---
*Next: [Evaluation](./Evaluation.md) for cross-industry benchmark results.*
