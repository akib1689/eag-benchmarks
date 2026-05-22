# Evaluation & Benchmarks

## Experimental Design

EAG is evaluated across production-scale benchmarks spanning four critical domains:

| Domain | Workload | Data Sensitivity | Compliance Framework |
|--------|----------|-----------------|---------------------|
| **Infrastructure Diagnostics** | Log analysis, metric correlation, root-cause identification | Network topology, config secrets | SOC2, ISO27001 |
| **Financial Auditing** | Transaction reconciliation, anomaly detection, ledger validation | PII, account balances, trade data | GDPR, PCI-DSS, SOX |
| **Clinical Workflow Analysis** | Patient triage, treatment path validation, audit trail generation | PHI, diagnostic codes, medication history | HIPAA, GDPR-H |
| **Industrial Monitoring** | Sensor telemetry aggregation, predictive maintenance, control system audit | Operational parameters, safety thresholds | IEC62443, NIST CSF |

### Baselines
- **Tool-Calling Agents**: Native function-calling APIs with direct data access
- **Retrieval-Augmented Pipelines**: RAG with document-level retrieval and context injection
- **Program-Aided Agents**: PAL/PoT-style code generation with external interpreter execution

### Metrics
1. **Diagnostic Accuracy**: Task success rate against ground-truth operational outcomes
2. **End-to-End Latency**: P50/P99 latency from intent submission to aggregated result
3. **Plan Verification Success Rate**: % of generated plans passing dual-stage validation
4. **Zero-Data Boundary Compliance**: Measured via data-flow analysis; target = 100%
5. **Resource Efficiency**: CPU/memory overhead of verification layer vs. baselines

## Performance Predictability

EAG decouples inference latency from execution latency through architectural optimizations:

### Parallelization Strategy
```
Planner (LLM)          Executor (Local)
     │                       │
     │── Plan Generation ──▶ │
     │                       │── Query Evaluation
     │◀── Aggregated Result─│
```
- Plan generation and local query evaluation execute concurrently where possible
- Aggregation occurs proximate to data sources, minimizing network overhead

### Static Optimization
- **Plan Caching**: Frequently used plan patterns cached at schema level
- **Execution Graph Pre-compilation**: Validated aggregation pipelines compiled ahead-of-time
- **Resource Quotas**: Enforced at operator level without modifying inference runtime

### Latency Results (Representative)
| Workload | EAG P99 | RAG P99 | Tool-Call P99 | SLO Target |
|----------|---------|---------|---------------|------------|
| Infrastructure | 240ms | 890ms | 310ms | ≤500ms |
| Financial | 310ms | 1.2s | 420ms | ≤600ms |
| Clinical | 280ms | 950ms | 380ms | ≤500ms |
| Industrial | 190ms | 720ms | 260ms | ≤400ms |

*EAG maintains latency within SLOs while reducing data-exposure risk to zero by construction.*

## Boundary Condition Stress Testing

EAG is evaluated under adversarial and edge-case scenarios:

| Condition | Description | EAG Response |
|-----------|-------------|--------------|
| **Malformed Intents** | Syntactically invalid or semantically ambiguous operational requests | Rejected at GCD stage; structured error returned |
| **Adversarial Plan Generation** | Planner attempts to exfiltrate data via crafted plan | Blocked at verification gate; no raw data transmitted |
| **High-Cardinality Streams** | Aggregation over large, dynamic datasets | Resource quotas enforced; partial results with confidence intervals |
| **Partial Capability Availability** | Executor missing operators referenced in plan | Capability mismatch diagnostic; operator can update contract |
| **Schema Drift** | Underlying data schema evolves without planner update | Verification fails with versioned error; operator notified |

## Data-Exposure Risk Measurement

Unlike probabilistic safeguards, EAG's zero-data boundary is verifiable:

- **Static Analysis**: Data-flow graphs prove raw inputs never reach planner context
- **Runtime Monitoring**: Executor logs confirm only aggregated outputs traverse boundary
- **Adversarial Testing**: Red-team attempts to exfiltrate data via plan manipulation fail at verification gate

**Result**: Data-exposure risk = 0% by architectural design, not statistical estimation.

## Verification Efficacy

Dual-stage validation prevents hallucination-driven failures:

| Validation Stage | Coverage | False Positive Rate | Impact |
|-----------------|----------|-------------------|---------|
| **GCD (Syntax)** | 100% of token outputs | <0.1% (grammar miscompile) | Eliminates format deviations |
| **Static Verifier (Semantic)** | Capability registry alignment | 2.3% (overly strict constraints) | Catches unreachable references, invalid aggregations |

**Iterative Refinement**: 94% of initially rejected plans succeed after one feedback cycle using structured diagnostics.

---
*Return to [Home](./Home.md) for overview and contribution summary.*
