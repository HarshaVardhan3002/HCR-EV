#!/usr/bin/env bash
# bootstrap.sh
# Recreates the full HCR-EV folder structure from scratch.
# Run once on a fresh clone or new machine.
# Does NOT create files, only directories.
# Files are committed to git and will exist after clone.

set -e

BASE="HCR-EV"

mkdir -p $BASE/data/raw
mkdir -p $BASE/data/processed
mkdir -p $BASE/data/synthetic/normal
mkdir -p $BASE/data/synthetic/anomaly
mkdir -p $BASE/data/exports
mkdir -p $BASE/models/l1/brakes
mkdir -p $BASE/models/l1/wheels
mkdir -p $BASE/models/l1/motor
mkdir -p $BASE/models/l1/battery
mkdir -p $BASE/models/l1/suspension_thermal
mkdir -p $BASE/models/l2
mkdir -p $BASE/models/l3/checkpoints
mkdir -p $BASE/models/l3/quantized
mkdir -p $BASE/models/l4
mkdir -p $BASE/routing/l2_normalizer
mkdir -p $BASE/routing/l3_helios/training
mkdir -p $BASE/routing/l3_helios/inference
mkdir -p $BASE/routing/l4_apex
mkdir -p $BASE/simulation/generators
mkdir -p $BASE/simulation/injectors
mkdir -p $BASE/simulation/scenarios
mkdir -p $BASE/training/configs
mkdir -p $BASE/training/pipelines
mkdir -p $BASE/training/experiments
mkdir -p $BASE/testing/unit
mkdir -p $BASE/testing/integration
mkdir -p $BASE/testing/benchmarks
mkdir -p $BASE/visualization/dashboards
mkdir -p $BASE/visualization/diagrams
mkdir -p $BASE/visualization/exports
mkdir -p $BASE/n8n/workflows
mkdir -p $BASE/n8n/configs
mkdir -p $BASE/brainstorm/architecture_iterations
mkdir -p $BASE/brainstorm/model_decisions
mkdir -p $BASE/brainstorm/diagrams
mkdir -p $BASE/brainstorm/notes
mkdir -p $BASE/docs/api
mkdir -p $BASE/docs/figures
mkdir -p $BASE/docs/specs
mkdir -p $BASE/assets

echo "HCR-EV directory structure created."
echo ""
echo "Next steps:"
echo "  1. cd HCR-EV"
echo "  2. git init (if new repo)"
echo "  3. Create ai_toolkit/ locally (gitignored, not in repo):"
echo "     mkdir -p ai_toolkit/prompts ai_toolkit/context ai_toolkit/settings ai_toolkit/checklists"
echo "     (copy files from your local backup or regenerate from context dump)"
echo "  4. pip install -r requirements.txt"
echo "  5. Verify ROCm: rocminfo | grep gfx   (expect: gfx1151)"
echo "  6. pytest testing/unit/ -v"
