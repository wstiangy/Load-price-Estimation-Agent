---
title: IEEE 14-Bus LMP Agent
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
short_description: IEEE 14-bus nodal pricing, inverse load estimation, sub-bus disaggregation, and probabilistic price forecasting.
suggested_hardware: cpu-basic
tags:
  - power-systems
  - optimal-power-flow
  - electricity-pricing
  - fastapi
  - research-demo
---

# IEEE 14-Bus LMP Agent

A research demo for electricity-price analysis on the IEEE 14-bus test system. The app runs a full workflow:

- DC optimal power flow with bus-level local marginal prices
- Inverse bus-load estimation from observed prices
- Sub-bus load disaggregation under each parent bus
- Sub-bus probabilistic load forecasting
- Probabilistic future LMP simulation by rerunning OPF on sampled load scenarios

## Local Run

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
$env:PYTHONPATH="src"
.\.venv\Scripts\python -m lmp_agent --output-dir outputs\run-1
.\.venv\Scripts\python -m uvicorn lmp_agent.dashboard:app --host 127.0.0.1 --port 8000
```

## Hugging Face Spaces

This repository is configured for a Docker Space.

1. Create a new Hugging Face Space and choose `Docker` as the SDK.
2. Push this repository to the Space.
3. Hugging Face will read the YAML block at the top of this README and build the included `Dockerfile`.
4. After the build finishes, open the Space URL and click `Run Default Scenario`.

The container serves the app on port `7860`, which matches the Space configuration.

## GitHub

The repo also includes:

- GitHub Actions CI in [ci.yml](C:\Users\guany\Dropbox\TAMU\Research\Load-price Estimation Agent\.github\workflows\ci.yml)
- Codespaces config in [devcontainer.json](C:\Users\guany\Dropbox\TAMU\Research\Load-price Estimation Agent\.devcontainer\devcontainer.json)

## Project Layout

- `src/lmp_agent/data.py`: IEEE 14-bus loader and synthetic scenario generator
- `src/lmp_agent/opf.py`: PyPSA-based DC-OPF and LMP extraction
- `src/lmp_agent/inverse.py`: hybrid surrogate plus regularized inverse optimization
- `src/lmp_agent/disaggregation.py`: constrained sub-bus allocation
- `src/lmp_agent/forecast.py`: quantile forecasting and probabilistic price simulation
- `src/lmp_agent/agent.py`: fixed workflow orchestrator
- `src/lmp_agent/dashboard.py`: FastAPI dashboard

## Notes

- The prototype uses a synthetic feeder schema because the standard IEEE 14-bus case does not include lower-voltage sub-buses.
- Probabilistic electricity prices are generated through load-scenario simulation and repeated OPF, not through a direct black-box price predictor.
