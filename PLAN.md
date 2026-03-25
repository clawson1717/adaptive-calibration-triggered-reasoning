# Adaptive Calibration-Triggered Reasoning (ACTR)

## Problem Statement

Current LLM reasoning systems either (a) apply a fixed reasoning strategy regardless of query difficulty, or (b) use uncertainty signals only for post-hoc rejection. Neither approach dynamically adjusts *how* the model reasons based on real-time uncertainty cues. Box Maze shows that boundary enforcement layers reduce failure rates to <1%, but doesn't continuously calibrate reasoning depth to the uncertainty of the current step. Hybrid uncertainty estimation (TwoSample + verbalized confidence) provides a calibrated uncertainty signal, but no system uses it as a real-time reasoning-mode controller.

This project combines Three-Sample Semantic Uncertainty (3-SSU) — which improves on the 2-sample hybrid by adding a third sample type for harder calibration — with Box Maze's layered process-control architecture, to create a system where **uncertainty signals directly trigger reasoning-mode switches**: fast/slow, shallow/deep verification, and knowledge-grounding intensity. Unlike reactive guardrails (RRG) that detect and correct *violations*, ACTR anticipates difficulty and *preemptively* adapts.

**Real problem solved:** In production LLM deployments, a single fixed reasoning strategy wastes compute on trivial queries while under-performing on hard ones. ACTR enables a compute-efficient adaptive reasoning system that concentrates expensive reasoning only where uncertainty is high.

---

## Architecture Overview

```
Query Input
    │
    ▼
┌──────────────────────────────────────────────────────┐
│         THREE-SAMPLE SEMANTIC UNCERTAINTY (3-SSU)    │
│  • Sample A: Standard temperature                    │
│  • Sample B: High temperature (+0.5)                 │
│  • Sample C: Contrastive decoding / alternative prompt│
│  → Semantic consistency score via embedding cosine   │
│  → Verbalized confidence regex extraction           │
│  → α-weighted fusion (learned α via calibration)      │
│  → Calibrated probability via temperature scaling    │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│         REASONING MODE CONTROLLER                     │
│  Threshold-based mode selection:                      │
│  • High confidence (p > 0.85): FAST mode            │
│  • Medium confidence (0.5 < p ≤ 0.85): MODERATE    │
│  • Low confidence (p ≤ 0.5): SLOW mode              │
│  Modes differ in: n_candidates, verification_depth,  │
│    knowledge_grounding, self-consistency_samples     │
└────────────────────────┬────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │    Mode-Specific Pipeline    │
          └──────────────┬──────────────┘
    ┌────────────────────┼────────────────────┐
    ▼                    ▼                    ▼
┌─────────┐        ┌────────────┐      ┌─────────────┐
│  FAST   │        │  MODERATE  │      │    SLOW     │
│  1 pass │        │  2 passes  │      │  3 passes   │
│  shallow│        │  standard  │      │  deep verif │
│  no KG  │        │  KG on fail │      │  full KG    │
└─────────┘        └────────────┘      └─────────────┘
    │                    │                    │
    └────────────────────┴────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│         BOUNDARY ENFORCEMENT LAYER (Box Maze)        │
│  • Rule-based boundary checks at each reasoning step │
│  • Memory grounding gate: only ground if p < 0.7    │
│  • Inference safety bounds: reject if p < 0.3       │
│  • Envelope: soft constraint propagation             │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
              ┌───────────────────┐
              │   Final Response  │
              │  + Confidence Tag  │
              │  + Reasoning Trace │
              └───────────────────┘
```

---

## 10-Step Implementation Plan

- [ ] **Step 1:** Project scaffolding — data structures, CLI interface, config, first-pass test suite
- [x] **Step 2:** Three-Sample Semantic Uncertainty Engine (3-SSU) — three sampler variants, embedding-based consistency, verbalized confidence regex, α-weighted fusion
- [x] **Step 3:** Temperature-scaled calibration — Platt scaling / temperature tuning on held-out calibration set
- [x] **Step 4:** Reasoning Mode Controller — threshold-based mode selector, mode transition logic
- [ ] **Step 5:** Fast Mode pipeline — single-pass generation, shallow heuristic check, confidence tag
- [x] **Step 6:** Moderate Mode pipeline — two-pass generation with conditional failure-triggered KG
- [x] **Step 7:** Slow Mode pipeline — three-pass with full knowledge grounding and deep verification
- [x] **Step 8:** Box Maze Boundary Enforcement Layer — memory grounding gate, inference safety bounds, envelope propagation
- [ ] **Step 9:** Integration + end-to-end tests, CLI `reason` command with confidence output, `benchmark` command
- [ ] **Step 10:** Calibration evaluation — measure accuracy vs. confidence calibration on held-out benchmarks

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Calibration Error (ECE) | < 0.05 | Expected Calibration Error on calibration set |
| Mode Switch Accuracy | > 85% | % of mode switches where difficulty estimate matches ground truth |
| Accuracy @ Fast Mode | > 90% | Accuracy on queries routed to Fast that deserved Fast |
| Accuracy @ Slow Mode | > 95% | Accuracy on queries routed to Slow that deserved Slow |
| Compute Efficiency | > 2× speedup | vs. always running Slow mode on all queries |
| Boundary Violation Rate | < 1% | % of responses violating safety bounds |
| Knowledge Grounding Precision | > 80% | % of KG activations that actually improve the response |

---

## Key Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | ≥ 3.10 | Core runtime |
| PyTorch | ≥ 2.0 | Sampling and embedding models |
| Transformers | ≥ 4.36 | LLM backbone + embeddings |
| OpenAI / Anthropic API | Latest | LLM inference |
| NumPy / SciPy | Latest | Calibration statistics |
| scikit-learn | Latest | Platt scaling, calibration |
| NetworkX | ≥ 3.0 | Reasoning graph representation |
| Weights & Biases | Latest | Experiment tracking |

---

## Prior Art References

1. **Box Maze** — Process-control architecture with memory grounding, structured inference, and boundary enforcement layers (<1% boundary failures)  
   *arXiv:2603.19182*

2. **Hybrid Uncertainty Estimation** — Two-sample hybrid combining self-consistency and verbalized confidence (AUROC +12 improvement)  
   *arXiv:2603.19118*

3. **OS-Themis** — Multi-agent critic with milestone decomposition and evidence chain auditing  
   *arXiv:2603.19191*

4. **RRG** — Reactive Reasoning Guardrails (our project): reactive guardrails with pattern detection (early pruning, path lock-in, boundary violation, knowledge prioritization)

5. **GUM** — Governed Uncertainty Memory (our project): Knowledge Objects + TwoSample AUROC + RPMS rule filtering + memory-layer hallucination prevention

6. **Contrastive Decoding** — Li et al., 2023 — alternative decoding strategy for diverse reasoning paths
