# brainstorm/model_decisions/l3_model_selection.md
# L3 HELIOS -- Model Evaluation and Selection Rationale

---

## Decision (Final)

Teacher: Qwen3.5-4B (MoE used for teacher only, no SSCR conflict at teacher stage)
Student: Gemma 3 1B, headless, dense architecture

Reason for Gemma 3: Dense transformer. Zero MoE gating. No architectural conflict
with Sparse Sequential Conditional Routing (SSCR), which is the thesis contribution.

---

## Models Evaluated and Rejected

### Nemotron 3 Nano 31.6B
- Status: REJECTED
- Reason: Server-class. Requires H200 or A100. Not edge-deployable.
- Target hardware (Jetson Orin NX) cannot run this model at any quantization.
- No further evaluation warranted.

### Nemotron 3 Nano 4B
- Status: REJECTED
- Reason: MoE backbone (mixture-of-experts gating).
- MoE gating mechanism directly conflicts with SSCR thesis contribution.
- Running SSCR on top of MoE creates architectural ambiguity: which routing
  mechanism is doing the work? Thesis claim collapses.
- Capability is adequate, hardware fit is fine. Architecture is the blocker.

### Qwen3.5-4B (MoE variant)
- Status: REJECTED for student role, ACCEPTED for teacher
- Reason (student): Same MoE conflict as Nemotron 4B.
- Reason (teacher): Teacher does not need to respect SSCR constraint. Teacher
  is distilled away. MoE in teacher is acceptable.
- Note: Qwen2.5-3B-Instruct used as placeholder until Qwen3.5-4B is on HF.

### Qwen3.5-2B dense
- Status: REJECTED
- Reason: Marginal reasoning capability for action schema output.
- Tested on synthetic action schema prompts: outputs lacked coherent
  constraint satisfaction (e.g., torque limits violated given RPM context).
- 1B gap to 3B makes meaningful difference at this task type.

---

## Alternatives Noted for L1

### LTFAD (All-MLP)
- Lighter than TCN. Validated on Jetson Xavier NX.
- Not selected for Phase 1. Documented here for thesis footnote.
- May revisit if TCN fails latency target on ESP32-S3.

### GRU-VAE (quantized)
- Alternative to LSTM+AE for suspension/thermal stream.
- Dynamic quantization approach. Reconstruction-based anomaly detection.
- Not selected. LSTM+AE held for now. Documented for comparison.

---

## Decision Date
Session 2, pre-departure sprint. Locked.
