#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
python -m lmp_agent --training-days "${TRAINING_DAYS:-6}" --monte-carlo-samples "${MONTE_CARLO_SAMPLES:-4}" --output-dir "${OUTPUT_DIR:-outputs/github-demo}"
