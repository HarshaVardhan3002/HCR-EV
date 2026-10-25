# HCR-EV: Hierarchical Control and Reasoning for Electric Vehicles

> A 4-layer edge AI pipeline for real-time EV powertrain control.
> Core contribution: Sparse Sequential Conditional Routing (SSCR).
> Status: design stage. The architecture and model selection are done; the code is in progress (see [Current State](#current-state-implemented)).

---

## What This Is

HCR-EV is a system architecture project. The primary deliverable is a reasoned,
validated design for an edge AI powertrain control pipeline -- not a production
deployment. The work demonstrates:

- System architecture design under hard latency and safety constraints (ISO 26262)
- Domain-specific model selection rationale with documented rejection trails
- Closed-loop synthetic simulation as a validation substitute for real hardware
- A new routing mechanism (SSCR) as the core contribution

---

## Full Vision (Not Current State)

When complete, HCR-EV will be a deployable edge AI stack for EV powertrain control
that reasons across all sensor domains simultaneously, adaptively, in real time,
with sub-200ms end-to-end latency on Jetson Orin NX-class hardware.

### Target Architecture

```
[L1] Sensor Expert Models
     TCN          -> Motor / Wheels / Brakes      (sub-20ms, Ambiq Apollo510 / ESP32-S3)
     TFT          -> Battery pack                 (sub-20ms)
     LSTM+AE      -> Suspension / Thermal          (sub-20ms)

[L2] Complexity Normalizer
     Stream fusion | Temporal alignment | Adaptive priority engine
     4 drive regimes: Highway / Urban / Dynamic / Stationary
     Static safety priority: Brakes(ASIL-D) > Wheels(ASIL-D) > Motor(ASIL-C)
                              > Battery(ASIL-C) > Suspension/Thermal(ASIL-B)

[L3] HELIOS (Language-grounded EV Intelligence and Output Sequencer)
     Qwen3.5-4B teacher -> Gemma 3 1B headless student (distillation)
     Custom action_schema_head: Linear(hidden_dim -> ACTION_SCHEMA_DIM)
     DPO on safe/unsafe action schema pairs
     AWQ INT4 quantized, target <150ms on Jetson Orin NX

[L4] APEX (Action Processing and Execution)
     Action dispatch | Deterministic brake safety layer | Hardware watchdog
     L4 -> L1 feedback: offline only (no runtime loop)
     L2/L3 circular dependency resolved via EMA of last 5 HELIOS priors
```

---

## Current State (Implemented)

| Component | Status |
|-----------|--------|
| Repo structure | Done |
| Architecture design (all 4 layers) | Done |
| Model selection + rejection rationale | Done |
| Synthetic data generator (Python) | In progress |
| L1 TCN skeleton (motor) | Planned -- Phase 1 |
| L2 normalizer stub | Planned -- Phase 1 |
| L3 HELIOS training pipeline | Planned -- Phase 2 |
| L4 APEX dispatch | Planned -- Phase 3 |
| Closed-loop simulation | Planned -- Phase 3 |
| arXiv preprint | Planned -- post Phase 1 |

---

## Execution Phases

### Phase 0 -- Foundation (Weeks 1-2, current)
- [x] Finalize architecture
- [x] Lock model decisions with documented rationale
- [ ] Push repo to GitHub
- [ ] Write synthetic data generator (normal + anomaly injection)
- [ ] ROCm verification on Strix Halo: `rocminfo | grep gfx` -> expect gfx1151

### Phase 1 -- L1 + L2 MVP (Weeks 3-6)
- Train TCN for motor sensor stream on synthetic data (RTX 4070)
- Stub L2 normalizer: static priority + EMA buffer (no ML yet)
- Latency benchmark: TCN inference time on RTX 4070
- Unit tests for data pipeline and normalizer logic

### Phase 2 -- HELIOS (Weeks 7-12)
- CPT + SFT Qwen3.5-4B on EV domain corpus (Strix Halo, bfloat16)
- Knowledge distillation into headless Gemma 3 1B
- DPO alignment on safe/unsafe action schema pairs
- AWQ INT4 quantization
- Latency benchmark: <150ms target

### Phase 3 -- Integration + Simulation (Weeks 13-18)
- Wire L1 -> L2 -> L3 -> L4 pipeline
- Closed-loop simulation: inject anomalies, verify routing + response
- APEX dispatch with deterministic brake override
- End-to-end latency benchmark: <200ms target

### Phase 4 -- Documentation + Publication (Weeks 19-24)
- arXiv preprint: SSCR contribution + architecture
- Industry-scoped proposal (withhold full HELIOS integration details)

---

## Synthetic Data Approach

No MATLAB/Simulink. Data is generated via Python randomizer scripts in
`simulation/generators/`. Normal operating ranges are derived from public EV
powertrain specs. Anomaly injection (`simulation/injectors/`) introduces
out-of-range values with configurable severity and duration.

Output format: `data/synthetic/normal/` and `data/synthetic/anomaly/` as
timestamped CSV + optional Parquet for training.

---

## Hardware

| Machine | Role |
|---------|------|
| ASUS ROG Strix Halo 395+ / 128GB unified | HELIOS teacher CPT+SFT, distillation, DPO, all heavy training |
| RTX 4070 8GB dGPU | L1 model training, AWQ INT4 quant, latency benchmarking |

Training cost: $0.00. Entire pipeline runs locally.

**Critical:** Use bfloat16 or Unsloth on Strix Halo. Gemma 3 activations overflow
float16 (max 65504). Verify ROCm first: `rocminfo | grep gfx` -> gfx1151.

---

## n8n Integration

`n8n/` contains workflow definitions for:
- Visualization pipeline: pipe model outputs to dashboards
- Debug flow: anomaly trigger -> alert -> log capture
- Experiment tracking: auto-log training run metadata

---

## Brainstorm Folder

`brainstorm/` is intentional. It documents architecture iterations, model
decisions (including rejections with rationale), and design reasoning. This is
not dead code -- it is the decision trail that justifies the final architecture.

Rejection log (summary):
- Nemotron 3 Nano 31.6B: server-class only (H200/A100), not edge-deployable
- Nemotron 3 Nano 4B: MoE backbone conflicts with the SSCR contribution
- Qwen3.5-4B MoE: same MoE conflict
- Qwen3.5-2B dense: marginal reasoning for action schema output

---

## Key Contribution

**Sparse Sequential Conditional Routing (SSCR):** A routing mechanism distinct
from standard MoE gating. Applied at L2 to L3 handoff. The complexity budget
invariant and closed-loop simulation validation pipeline are secondary
contributions.

Gemma 3 1B was selected specifically because its dense architecture avoids MoE
gating conflicts with SSCR. This is not a default choice -- it is an
architecture-driven decision.

---

## Repository Map

```
HCR-EV/
  data/              -- raw, processed, synthetic (normal + anomaly), exports
  models/            -- l1 per-sensor, l2, l3 (checkpoints + quantized), l4
  routing/           -- l2 normalizer, l3 helios, l4 apex implementations
  simulation/        -- generators, injectors, scenarios
  training/          -- configs, pipelines, experiments
  testing/           -- unit, integration, benchmarks
  visualization/     -- dashboards, diagrams, exports
  n8n/               -- workflow definitions and configs
  brainstorm/        -- architecture iterations, model decisions, notes
  docs/              -- api, figures, specs
  assets/            -- static files
  ai_toolkit/        -- LOCAL ONLY, gitignored
```

---

## Contact

Harsha Vardhan | OvGU Magdeburg M.Sc. Digital Engineering (SuSe 2026)
GitHub: HarshaVardhan3002
