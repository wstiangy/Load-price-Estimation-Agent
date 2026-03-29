#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
python -m uvicorn lmp_agent.dashboard:app --host 0.0.0.0 --port 8000
