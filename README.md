# Adaptive Calibration-Triggered Reasoning (ACTR)

A reasoning system where **uncertainty signals directly trigger reasoning-mode switches** — fast, moderate, or slow — based on real-time calibrated confidence estimates. Combines Three-Sample Semantic Uncertainty (3-SSU) with Box Maze's layered process-control architecture.

## Core Idea

Most LLM systems apply a fixed reasoning strategy regardless of query difficulty. ACTR measures semantic uncertainty at each reasoning step using a three-sample hybrid (standard, high-temperature, and contrastive-decoding samples), calibrates those estimates, and uses the calibrated confidence to decide *how deeply* to reason. Trivial queries get fast single-pass answers; hard queries trigger multi-pass verification with knowledge grounding.

## Status

- [x] Project planned
- [x] Step 1: Project scaffolding and data structures
- [x] Step 2: Three-Sample Semantic Uncertainty Engine (3-SSU)
- [ ] Step 3: Temperature-scaled calibration
- [ ] Step 4: Reasoning Mode Controller
- [ ] Step 5: Fast Mode pipeline
- [ ] Step 6: Moderate Mode pipeline
- [ ] Step 7: Slow Mode pipeline
- [ ] Step 8: Box Maze Boundary Enforcement Layer
- [ ] Step 9: Integration + end-to-end tests
- [ ] Step 10: Calibration evaluation

## Architecture

```
Query
  ↓
3-SSU Uncertainty Estimator (3 samples → calibrated p)
  ↓
Reasoning Mode Controller (threshold-based routing)
  ├─ High confidence (p > 0.85) → Fast Mode (1 pass)
  ├─ Medium confidence (0.5 < p ≤ 0.85) → Moderate Mode (2 passes)
  └─ Low confidence (p ≤ 0.5) → Slow Mode (3 passes + KG)
  ↓
Box Maze Boundary Enforcement Layer
  ↓
Final Response + Confidence Tag + Reasoning Trace
```

## CLI

```bash
# Query with confidence output
python3 -m src.cli reason "What is the capital of France?" --verbose

# Benchmark on a dataset
python3 -m src.cli benchmark --dataset questions.jsonl --output results.json

# Run calibration evaluation
python3 -m src.cli calibrate --calibration-set cal_set.jsonl --model gpt-4

# Evaluate reasoning mode routing accuracy
python3 -m src.cli eval-routing --test-set test.jsonl
```

## Key Techniques

- **3-SSU**: Three-sample semantic uncertainty (standard + high-temperature + contrastive-decoding)
- **Platt Scaling**: Temperature-tuned calibration of confidence scores
- **Box Maze Boundary Enforcement**: Memory grounding gate, inference safety bounds, envelope propagation
- **Adaptive Compute**: Mode switches reduce compute on easy queries, increase depth on hard ones

## Prior Art

- Box Maze (arXiv:2603.19182) — boundary enforcement layers
- Hybrid Uncertainty Estimation (arXiv:2603.19118) — 2-sample hybrid AUROC
- OS-Themis (arXiv:2603.19191) — milestone decomposition + evidence chains
- Contrastive Decoding (Li et al., 2023) — alternative reasoning paths
