# Related Work

## Separation of Program and Execution

### PAL: Program-Aided Language Models [4]
**Approach**: LLM emits Python code as intermediate steps; external interpreter handles computation.
**Strengths**: 
- Deterministic computation offload eliminates arithmetic hallucinations
- Prompt-only adaptation; no fine-tuning required
- Semantic variable grounding improves code fidelity
**Gap vs. EAG**: 
- Assumes trusted, unsandboxed execution environment
- Turing-complete output prevents static safety guarantees
- No data boundary enforcement; raw problem data flows into execution context
**EAG Extension**: Operates under stricter threat model: execution occurs in read-only, schema-constrained parser with zero raw-data exposure to LLM.

### Program of Thoughts (PoT) [2]
**Approach**: Decouples reasoning from computation via executable code generation.
**Strengths**: 
- Deterministic arithmetic via external runtime
- Zero-shot performance gains on numerical benchmarks
**Gap vs. EAG**: 
- Trusted execution assumption with no data isolation
- Arbitrary code generation complicates safety verification
**EAG Extension**: Extends separation principle into security domain with schema-constrained plans.

### ReAct: Reasoning + Acting [15]
**Approach**: Interleaves thoughts and actions in a single loop; grounds reasoning in external observations.
**Strengths**: 
- Reduces factual drift via real-world grounding
- Flexible tool integration without retraining
- Interpretable execution traces
**Gap vs. EAG**: 
- Open-loop execution; tool calls bypass pre-flight validation
- Trusted execution assumption; no isolation or data filtering
- Reactive error handling vs. proactive verification
**EAG Extension**: Retains reasoning-action synergy but decouples planning from execution with data isolation guarantees.

### DSPy: Declarative LM Pipeline Compilation [6]
**Approach**: Compiles declarative specifications into optimized prompt pipelines via teleprompters.
**Strengths**: 
- Declarative abstraction decouples what from how
- Metric-driven optimization reduces manual prompt engineering
- Composable control flow bridges neural and symbolic paradigms
**Gap vs. EAG**: 
- Raw inputs flow directly into LM prompts; no read-only enforcement
- Execution semantics opaque; no schema-guided validation
- Optimization limited to prompt space, not computational safety
**EAG Extension**: Compiles schema-constrained execution plans validated by local, read-only parser; enforces zero raw-data exposure at inference time.

## Isolation Execution & Data Privacy

### CMIF: Confidential LLM Inference [17]
**Approach**: Hybrid privacy via client-side TEEs + differential privacy for embedding layer.
**Strengths**: 
- Defense-in-depth via TEE + DP composition
- Semantic-aware sanitization preserves utility
- Practical partitioning minimizes enclave-GPU roundtrips
**Gap vs. EAG**: 
- Input-centric protection; sanitized tokens still flow into LLM computation graph
- No output-side guarantees against inference attacks
- TEE-centric trust assumption vulnerable to side-channel attacks
**EAG Extension**: Enforces stricter invariant: zero raw-data exposure at any stage. Compiles intent into schema-guided plans evaluated locally.

### STELP: Secure Transpilation of LLM Code [10]
**Approach**: Runtime safety layer intercepts LLM-generated code, validates against safe grammar subset, transpiles with embedded controls.
**Strengths**: 
- Policy-driven transpilation enables fine-grained safety policies
- InjectedHumanEval benchmark for joint correctness/security evaluation
- Self-repair feedback loop reduces manual intervention
**Gap vs. EAG**: 
- Validates syntactic structure, not semantic intent
- Raw input data still flows into execution context
- No schema-constrained data access enforcement
**EAG Extension**: Operates at higher abstraction: validates schema-guided execution plans where raw user data never leaves local parser.

### DP-RAG: Privacy-Preserving Retrieval [7]
**Approach**: Formal differential privacy for RAG via private voting over document shards.
**Strengths**: 
- First formal DP formulation for RAG with (ε,δ)-guarantees
- Sparse budgeting via disagreement detection improves utility
- Modular, model-agnostic design
**Gap vs. EAG**: 
- Retrieval step excluded from DP guarantee
- Assumes LLM is trusted processor of sanitized inputs
- Per-query composition overhead complicates multi-turn privacy accounting
**EAG Extension**: Inverts assumption: instead of accounting for exposure via ε, enforces zero raw-data exposure by design.

## DSL, Constrained Decoding & Verification

### Grammar-Constrained Decoding (GCD) [5]
**Approach**: Incremental parser filters LM token distribution to retain only grammar-valid continuations.
**Strengths**: 
- Declarative structure specification via human-readable CFGs
- Input-adaptive constraints reduce search space
- Zero-shot structured reliability without fine-tuning
**Gap vs. EAG**: 
- Requires logit access; excludes closed API models
- Grammar design overhead demands formal language expertise
- Enforces syntactic validity only, not semantic correctness
**EAG Integration**: EAG adopts GCD as core enforcement mechanism: schema compiler translates execution-plan specs into input-dependent CFGs, guaranteeing only syntactically valid, schema-compliant plans are emitted.

### NeMo Guardrails [9]
**Approach**: Runtime toolkit for programmable, interpretable guardrails via Colang specifications.
**Strengths**: 
- Runtime programmability without model retraining
- Interpretable control flow via explicit dialogue policies
- Composable safety layers for fine-grained control
**Gap vs. EAG**: 
- Sequential latency overhead from 3-step prompting chain
- Raw data exposure; no read-only or schema-constrained handling
- Prompt-dependent robustness vulnerable to adversarial perturbations
**EAG Integration**: Adopts decoupled-control philosophy but under stricter invariant: zero raw-data exposure. Guardrails can validate policy constraints on aggregated results, not raw logs.

### SandboxEval [8]
**Approach**: Empirical benchmark suite for validating security boundaries of sandboxed execution environments.
**Strengths**: 
- Empirical over declarative validation; measures actual behavior
- Principle-of-least-privilege test design enables precise attribution
- Portable, language-agnostic threat taxonomy
**Gap vs. EAG**: 
- Coverage limitations; cannot guarantee completeness
- Static test corpus; no adaptive fuzzing
- Assumes well-formed code; no handling of malformed LLM outputs
**EAG Integration**: Complementary layered defense: EAG prevents prohibited operations at plan level; SandboxEval validates runtime isolation. Recommend running SandboxEval against EAG executor as CI check.

---
*Next: [Evaluation](./Evaluation.md) for experimental results and benchmarks.*
