# Contributing to HCR-EV

This is a solo research project (Master's thesis, OvGU Magdeburg).
External contributions are not accepted at this stage.

If you are a supervisor, collaborator, or industry reviewer reading this:
architecture decisions and implementation rationale are documented in brainstorm/.
The model rejection log is at brainstorm/model_decisions/l3_model_selection.md.

---

## For Harsha (Self Reference)

### Before committing

Run through ai_toolkit/checklists/pre_commit_checklist.md.
All tests must pass: pytest testing/

### Branching

main: stable, phase-complete code only
dev: active development
feature/phase-N-description: per-phase feature branches

### When architecture changes

1. Update brainstorm/model_decisions/ with the decision and rationale.
2. Update ai_toolkit/context/architecture_dump.md.
3. Update README.md Current State table.
4. Update ai_toolkit/context/phase_status.md.

Do not change architecture silently. The decision trail is part of the thesis.
