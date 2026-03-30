from __future__ import annotations

import json
import os
from html import escape
from typing import Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .agent import PricingWorkflowAgent
from .chat import generate_chat_reply, llm_status
from .config import RunConfig, WorkflowArtifacts

app = FastAPI(title="IEEE 14-Bus Pricing Agent Dashboard")

BUS_POSITIONS: dict[str, tuple[float, float]] = {
    "Bus 1": (0.08, 0.52),
    "Bus 2": (0.22, 0.52),
    "Bus 3": (0.36, 0.60),
    "Bus 4": (0.36, 0.42),
    "Bus 5": (0.22, 0.28),
    "Bus 6": (0.40, 0.14),
    "Bus 7": (0.58, 0.52),
    "Bus 8": (0.74, 0.52),
    "Bus 9": (0.58, 0.34),
    "Bus 10": (0.74, 0.34),
    "Bus 11": (0.88, 0.34),
    "Bus 12": (0.58, 0.10),
    "Bus 13": (0.74, 0.10),
    "Bus 14": (0.90, 0.18),
}

GENERATOR_POSITIONS: dict[str, tuple[float, float]] = {
    "G1": (0.03, 0.60),
    "G2": (0.18, 0.64),
    "G3": (0.41, 0.70),
    "G4": (0.31, 0.03),
    "G5": (0.79, 0.62),
}

EDGE_LIST = [
    ("Bus 1", "Bus 2"),
    ("Bus 1", "Bus 5"),
    ("Bus 2", "Bus 3"),
    ("Bus 2", "Bus 4"),
    ("Bus 2", "Bus 5"),
    ("Bus 3", "Bus 4"),
    ("Bus 4", "Bus 5"),
    ("Bus 4", "Bus 7"),
    ("Bus 4", "Bus 9"),
    ("Bus 5", "Bus 6"),
    ("Bus 6", "Bus 11"),
    ("Bus 6", "Bus 12"),
    ("Bus 6", "Bus 13"),
    ("Bus 7", "Bus 8"),
    ("Bus 7", "Bus 9"),
    ("Bus 9", "Bus 10"),
    ("Bus 9", "Bus 14"),
    ("Bus 10", "Bus 11"),
    ("Bus 12", "Bus 13"),
    ("Bus 13", "Bus 14"),
]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    context: dict[str, Any]
    previous_response_id: str | None = Field(default=None, max_length=200)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
    <html>
      <head>
        <title>IEEE 14-Bus Pricing Agent</title>
        <style>
          :root {
            --ink: #10233e;
            --muted: #50627d;
            --paper: rgba(255,255,255,0.88);
            --accent: #b2492f;
            --line: rgba(16,35,62,0.12);
          }
          * { box-sizing: border-box; }
          body {
            margin: 0;
            color: var(--ink);
            font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif;
            background:
              radial-gradient(circle at top left, rgba(233,181,106,0.22), transparent 28%),
              radial-gradient(circle at top right, rgba(178,73,47,0.18), transparent 32%),
              linear-gradient(180deg, #f6f1e7 0%, #edf3f7 52%, #eef0e7 100%);
          }
          .page { max-width: 1160px; margin: 0 auto; padding: 34px 22px 48px; }
          .hero {
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 28px 30px;
            background: linear-gradient(130deg, rgba(255,255,255,0.96), rgba(255,247,235,0.88));
            box-shadow: 0 24px 60px rgba(16,35,62,0.10);
          }
          .eyebrow {
            display: inline-block;
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(178,73,47,0.10);
            color: var(--accent);
            font-size: 12px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
          }
          h1 { margin: 16px 0 12px; font-size: clamp(2.3rem, 5vw, 4.2rem); line-height: 0.95; }
          p { margin: 0 0 14px; font-size: 1.05rem; line-height: 1.65; color: var(--muted); }
          .hero-grid, .steps { display: grid; gap: 18px; }
          .hero-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          .steps { grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 20px; }
          .panel, .step {
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 18px 18px 16px;
            background: rgba(255,255,255,0.78);
          }
          .step strong, .panel strong {
            display: block;
            margin-bottom: 8px;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--accent);
          }
          .actions { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 20px; }
          a.button {
            display: inline-block;
            padding: 12px 18px;
            border-radius: 999px;
            background: linear-gradient(135deg, #17395c, #315d76);
            color: white;
            text-decoration: none;
            box-shadow: 0 14px 24px rgba(23,57,92,0.22);
          }
          .ghost { background: rgba(16,35,62,0.06); color: var(--ink); box-shadow: none; }
          code { background: rgba(16,61,92,0.08); padding: 0.15rem 0.35rem; border-radius: 6px; }
          @media (max-width: 860px) { .hero-grid, .steps { grid-template-columns: 1fr; } }
        </style>
      </head>
      <body>
        <div class="page">
          <section class="hero">
            <span class="eyebrow">IEEE 14-Bus Electricity Market Demo</span>
            <h1>Load, Price, and Topology in One Research Dashboard</h1>
            <div class="hero-grid">
              <div>
                <p>This demo runs a full fixed workflow: DC-OPF, price-to-load inversion, sub-bus disaggregation, probabilistic load forecasting, and probabilistic LMP propagation.</p>
                <p>Use the results page to inspect the IEEE 14-bus topology with bus-level load and price labels placed directly next to each node.</p>
              </div>
              <div class="panel">
                <strong>Quick Run</strong>
                <p>Default experiment: <code>/run</code></p>
                <p>Custom example: <code>/run?seed=11&training_days=10&demand_scale=1.1&noise_scale=0.1&focus_hour=18</code></p>
                <p>Structured output: <code>/api/run</code></p>
              </div>
            </div>
            <div class="actions">
              <a class="button" href="/run">Run Default Scenario</a>
              <a class="button ghost" href="/run?seed=11&training_days=10&demand_scale=1.1&noise_scale=0.1&focus_hour=18">Open Evening Stress Case</a>
            </div>
            <div class="steps">
              <div class="step">
                <strong>1. OPF</strong>
                <p>Compute nodal marginal prices under the IEEE 14-bus transmission constraints.</p>
              </div>
              <div class="step">
                <strong>2. Inference</strong>
                <p>Invert observed LMPs into parent-bus demand and split demand into synthetic sub-buses.</p>
              </div>
              <div class="step">
                <strong>3. Forecast</strong>
                <p>Generate probabilistic sub-bus load trajectories and propagate them to future prices.</p>
              </div>
            </div>
          </section>
        </div>
      </body>
    </html>
    """


@app.get("/api/run", response_class=JSONResponse)
def api_run(
    seed: int = 7,
    training_days: int = 16,
    demand_scale: float = 1.0,
    noise_scale: float = 0.08,
    monte_carlo_samples: int = 20,
) -> dict[str, Any]:
    config = RunConfig(
        seed=seed,
        training_days=training_days,
        demand_scale=demand_scale,
        noise_scale=noise_scale,
        monte_carlo_samples=monte_carlo_samples,
    )
    agent = PricingWorkflowAgent(config)
    artifacts = agent.run_pipeline()
    return {
        "metrics": artifacts.metrics,
        "current_opf_feasible": artifacts.current_opf.feasible,
        "plan": agent.plan_run(),
    }


@app.post("/api/chat", response_class=JSONResponse)
def api_chat(request: ChatRequest) -> dict[str, Any]:
    try:
        return generate_chat_reply(
            user_message=request.message,
            page_context=request.context,
            previous_response_id=request.previous_response_id,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/run", response_class=HTMLResponse)
def run_page(
    seed: int = 7,
    training_days: int = 16,
    demand_scale: float = 1.0,
    noise_scale: float = 0.08,
    monte_carlo_samples: int = 20,
    focus_hour: int | None = None,
) -> str:
    config = RunConfig(
        seed=seed,
        training_days=training_days,
        demand_scale=demand_scale,
        noise_scale=noise_scale,
        monte_carlo_samples=monte_carlo_samples,
    )
    artifacts = PricingWorkflowAgent(config).run_pipeline()
    focus_hour = _resolve_focus_hour(artifacts, focus_hour)

    topology_plot = _topology_plot_html(artifacts, focus_hour)
    load_plot = _load_comparison_plot_html(
        artifacts.current_truth_bus_load,
        artifacts.bus_estimate.estimated_bus_load,
        focus_hour,
    )
    subbus_plot = _subbus_quantile_plot_html(artifacts.forecast.subbus_load_quantiles)
    price_plot = _price_quantile_plot_html(artifacts.forecast.bus_lmp_quantiles, focus_hour)
    summary_cards = _summary_cards_html(artifacts, focus_hour)
    highlights = _highlights_html(artifacts, focus_hour)
    bus_table = _bus_snapshot_table_html(artifacts, focus_hour)
    generator_table = _generator_snapshot_table_html(artifacts, focus_hour)
    line_table = _line_snapshot_table_html(artifacts, focus_hour)
    pipeline_strip = _pipeline_strip_html()
    chat_panel = _llm_panel_html(artifacts, focus_hour)
    report_html = escape(artifacts.report_markdown)

    return f"""
    <html>
      <head>
        <title>IEEE 14-Bus Research Demo</title>
        <style>
          :root {{
            --ink: #12253d; --muted: #5f6f87; --panel: rgba(255,255,255,0.88);
            --accent: #b24a30; --accent-deep: #7d2b1c; --teal: #2f6f71;
            --line: rgba(18,37,61,0.10); --shadow: 0 26px 58px rgba(18,37,61,0.10);
          }}
          * {{ box-sizing: border-box; }}
          body {{
            margin: 0; color: var(--ink);
            font-family: "Palatino Linotype", "Book Antiqua", Palatino, serif;
            background:
              radial-gradient(circle at top left, rgba(201,146,46,0.18), transparent 26%),
              radial-gradient(circle at top right, rgba(47,111,113,0.16), transparent 30%),
              linear-gradient(180deg, #f7f1e6 0%, #eff4f7 56%, #eef0e4 100%);
          }}
          .page {{ max-width: 1320px; margin: 0 auto; padding: 28px 20px 44px; }}
          .hero {{
            position: relative; overflow: hidden; border: 1px solid var(--line); border-radius: 28px;
            background: linear-gradient(128deg, rgba(255,255,255,0.96), rgba(255,246,233,0.90));
            box-shadow: var(--shadow); padding: 28px 30px 26px;
          }}
          .hero::after {{
            content: ""; position: absolute; width: 280px; height: 280px; right: -70px; top: -120px;
            border-radius: 50%; background: radial-gradient(circle, rgba(178,74,48,0.15), transparent 68%);
          }}
          .eyebrow {{
            display: inline-block; border-radius: 999px; padding: 8px 12px; background: rgba(178,74,48,0.10);
            color: var(--accent); font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase;
          }}
          .hero-grid {{ display: grid; grid-template-columns: 1.4fr 0.95fr; gap: 18px; position: relative; z-index: 1; }}
          h1 {{ margin: 16px 0 12px; font-size: clamp(2.1rem, 4.8vw, 4.3rem); line-height: 0.94; }}
          h2 {{ margin: 0 0 12px; font-size: 1.45rem; }}
          p {{ margin: 0 0 12px; line-height: 1.68; color: var(--muted); font-size: 1.03rem; }}
          .run-meta, .cta-row {{ display: flex; flex-wrap: wrap; gap: 10px; }}
          .run-meta {{ margin-top: 16px; }}
          .cta-row {{ gap: 12px; margin-top: 18px; }}
          .chip {{
            border-radius: 999px; padding: 8px 12px; background: rgba(18,37,61,0.06); color: var(--ink); font-size: 0.95rem;
          }}
          a.button {{
            display: inline-block; padding: 12px 18px; border-radius: 999px; text-decoration: none;
            background: linear-gradient(135deg, #17395c, #2f6f71); color: white; box-shadow: 0 14px 24px rgba(23,57,92,0.18);
          }}
          a.button.alt {{ background: rgba(18,37,61,0.06); color: var(--ink); box-shadow: none; }}
          .insight-panel, .panel {{
            border: 1px solid var(--line); border-radius: 22px; background: var(--panel); box-shadow: var(--shadow);
            padding: 20px 20px 18px;
          }}
          .insight-panel {{ background: linear-gradient(180deg, rgba(255,251,246,0.96), rgba(249,252,255,0.92)); }}
          .stats, .pipeline, .secondary-grid {{ display: grid; gap: 14px; }}
          .stats {{ grid-template-columns: repeat(4, minmax(0, 1fr)); margin: 20px 0 18px; }}
          .pipeline {{ grid-template-columns: repeat(4, minmax(0, 1fr)); margin: 0 0 18px; }}
          .secondary-grid {{ grid-template-columns: 1fr 1fr; margin-top: 18px; gap: 18px; }}
          .stat, .stage {{
            border: 1px solid var(--line); border-radius: 18px; padding: 16px 16px 14px; background: rgba(255,255,255,0.80);
          }}
          .stage {{ background: rgba(18,37,61,0.05); padding: 15px 15px 13px; }}
          .stat span, .stage strong, .section-label {{
            display: inline-block; margin-bottom: 8px; font-size: 0.82rem; letter-spacing: 0.08em;
            text-transform: uppercase; color: var(--accent);
          }}
          .stage strong {{ display: block; color: var(--accent-deep); }}
          .stat strong {{ display: block; font-size: 1.7rem; line-height: 1.05; }}
          .layout {{ display: grid; grid-template-columns: 1.34fr 0.86fr; gap: 18px; align-items: start; }}
          .stack {{ display: grid; gap: 18px; }}
          table {{ width: 100%; border-collapse: collapse; font-size: 0.95rem; }}
          th, td {{ padding: 10px 8px; border-bottom: 1px solid rgba(18,37,61,0.08); text-align: left; }}
          th {{ color: var(--accent-deep); font-size: 0.83rem; text-transform: uppercase; letter-spacing: 0.06em; }}
          .note {{
            margin-top: 12px; padding: 12px 14px; border-radius: 14px; background: rgba(178,74,48,0.06); color: var(--muted);
          }}
          .chat-shell {{
            display: grid; gap: 12px;
          }}
          .chat-transcript {{
            min-height: 260px; max-height: 420px; overflow-y: auto; padding: 14px;
            border: 1px solid rgba(18,37,61,0.08); border-radius: 16px; background: rgba(247,250,252,0.88);
          }}
          .chat-message {{
            margin-bottom: 12px; padding: 12px 13px; border-radius: 14px; line-height: 1.55;
          }}
          .chat-message strong {{
            display: block; margin-bottom: 6px; font-size: 0.82rem; letter-spacing: 0.06em; text-transform: uppercase;
          }}
          .chat-message.user {{ background: rgba(23,57,92,0.08); }}
          .chat-message.assistant {{ background: rgba(47,111,113,0.10); }}
          .chat-message.status {{ background: rgba(178,74,48,0.08); color: var(--muted); }}
          .chat-controls {{ display: grid; gap: 10px; }}
          .chat-controls textarea {{
            width: 100%; min-height: 104px; resize: vertical; padding: 12px 14px;
            border-radius: 16px; border: 1px solid rgba(18,37,61,0.12); background: rgba(255,255,255,0.94);
            color: var(--ink); font: inherit;
          }}
          .chat-actions {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
          .chat-send {{
            border: 0; cursor: pointer; padding: 11px 16px; border-radius: 999px;
            background: linear-gradient(135deg, #17395c, #2f6f71); color: white; box-shadow: 0 12px 22px rgba(23,57,92,0.18);
            font: inherit;
          }}
          .chat-send[disabled] {{ opacity: 0.55; cursor: wait; }}
          .chat-hint {{ color: var(--muted); font-size: 0.94rem; }}
          .chat-suggestions {{ display: flex; flex-wrap: wrap; gap: 8px; }}
          .chat-chip {{
            border: 1px solid rgba(18,37,61,0.10); background: rgba(18,37,61,0.05); color: var(--ink);
            border-radius: 999px; padding: 8px 12px; cursor: pointer; font: inherit;
          }}
          .chat-chip[disabled] {{ cursor: default; opacity: 0.65; }}
          pre {{ margin: 0; white-space: pre-wrap; font-size: 0.95rem; line-height: 1.55; }}
          @media (max-width: 1080px) {{
            .hero-grid, .layout, .secondary-grid, .stats, .pipeline {{ grid-template-columns: 1fr; }}
          }}
        </style>
      </head>
      <body>
        <div class="page">
          <section class="hero">
            <span class="eyebrow">IEEE 14-Bus Market Analysis</span>
            <div class="hero-grid">
              <div>
                <h1>Topology-Centered Price and Load Inference Demo</h1>
                <p>This page is structured like a compact paper demo: the topology view is the centerpiece, the workflow is summarized explicitly, and the single-line topology is annotated with generator dispatch, marginal costs, nodal LMPs, line flows, and congestion headroom.</p>
                <p>The highlighted snapshot below uses <strong>hour {focus_hour:02d}</strong>. Generator callouts show dispatch and cost, load buses show LMP, and each line segment displays its MW flow together with the remaining margin before congestion.</p>
                <div class="run-meta">
                  <span class="chip">seed={seed}</span>
                  <span class="chip">training_days={training_days}</span>
                  <span class="chip">demand_scale={demand_scale:.2f}</span>
                  <span class="chip">noise_scale={noise_scale:.2f}</span>
                  <span class="chip">MC samples={monte_carlo_samples}</span>
                </div>
                <div class="cta-row">
                  <a class="button" href="/">Back To Landing Page</a>
                  <a class="button alt" href="/run?seed={seed}&training_days={training_days}&demand_scale={demand_scale}&noise_scale={noise_scale}&monte_carlo_samples={monte_carlo_samples}&focus_hour={max(focus_hour - 1, 0)}">Compare Nearby Hour</a>
                </div>
              </div>
              <div class="insight-panel">
                <span class="section-label">Research Framing</span>
                <h2>What This Run Demonstrates</h2>
                {highlights}
              </div>
            </div>
          </section>
          <section class="stats">
            {summary_cards}
          </section>
          <section class="pipeline">
            {pipeline_strip}
          </section>
          <section class="layout">
            <div class="stack">
              <div class="panel">
                <span class="section-label">Topology Snapshot</span>
                <h2>IEEE 14-Bus Single-Line Topology With Generator, LMP, and Line Labels</h2>
                <p>The topology plot below uses a fixed IEEE 14-bus single-line layout so that the labels remain spatially stable and close to the reference IEEE 14-bus diagram.</p>
                {topology_plot}
                <div class="note">Warmer lines are closer to congestion. Generator callouts show dispatch and marginal cost, bus callouts show LMP, and line callouts show signed MW flow with remaining thermal headroom.</div>
              </div>
              <div class="secondary-grid">
                <div class="panel">
                  <span class="section-label">Load Reconstruction</span>
                  <h2>Truth vs Inferred Bus Load</h2>
                  <p>Dashed lines show the inverse-estimation output reconstructed from nodal prices and prior load structure.</p>
                  {load_plot}
                </div>
                <div class="panel">
                  <span class="section-label">Probabilistic Price View</span>
                  <h2>Future LMP Quantile Trajectories</h2>
                  <p>P10-P90 bands are generated by simulating forecasted sub-bus demand through repeated OPF runs.</p>
                  {price_plot}
                </div>
              </div>
            </div>
            <div class="stack">
              <div class="panel">
                <span class="section-label">LLM Copilot</span>
                <h2>Interactive Research Assistant</h2>
                <p>Ask the page to interpret the current topology snapshot, explain a bus-level price pattern, or summarize forecast uncertainty in plain language.</p>
                {chat_panel}
              </div>
              <div class="panel">
                <span class="section-label">Generator Snapshot</span>
                <h2>Generator Dispatch and Marginal Cost</h2>
                <p>Each online generator is mapped to its parent bus and shown using the same hour selected in the topology snapshot.</p>
                {generator_table}
              </div>
              <div class="panel">
                <span class="section-label">Line Monitor</span>
                <h2>Line Loading and Congestion Margin</h2>
                <p>Headroom is computed as thermal limit minus absolute flow, so near-zero values indicate lines approaching congestion.</p>
                {line_table}
              </div>
              <div class="panel">
                <span class="section-label">Bus Snapshot</span>
                <h2>Bus-Level Load and Price Table</h2>
                <p>The selected hour is rendered explicitly so that the topology labels and numeric table stay aligned.</p>
                {bus_table}
              </div>
              <div class="panel">
                <span class="section-label">Sub-Bus Forecast</span>
                <h2>Sub-Bus Quantile Fan</h2>
                <p>The first few synthetic sub-buses are shown here to illustrate how the distribution-level load shapes widen into forecast bands.</p>
                {subbus_plot}
              </div>
              <div class="panel">
                <span class="section-label">Machine Report</span>
                <h2>Experiment Summary</h2>
                <pre>{report_html}</pre>
              </div>
            </div>
          </section>
        </div>
      </body>
    </html>
    """


def _resolve_focus_hour(artifacts: WorkflowArtifacts, focus_hour: int | None) -> int:
    if focus_hour is None:
        return int(artifacts.current_truth_bus_load.sum(axis=1).idxmax())
    return max(0, min(23, int(focus_hour)))


def _summary_cards_html(artifacts: WorkflowArtifacts, focus_hour: int) -> str:
    focus_lmp = artifacts.current_opf.bus_lmp.loc[focus_hour]
    focus_load = artifacts.current_truth_bus_load.loc[focus_hour]
    top_price_bus = focus_lmp.idxmax()
    top_load_bus = focus_load.idxmax()
    median_q = sorted(artifacts.config.quantiles)[len(artifacts.config.quantiles) // 2]
    future_lmp = artifacts.forecast.bus_lmp_quantiles[median_q].loc[focus_hour]
    cards = [
        ("Peak Snapshot Demand", f"{focus_load.sum():.1f} MW", f"{top_load_bus} carries the largest bus load at hour {focus_hour:02d}."),
        ("Highest Snapshot LMP", f"{focus_lmp.max():.2f}", f"{top_price_bus} is the highest-priced node at the selected hour."),
        ("Inverse Bus MAPE", f"{artifacts.metrics['inverse_bus_mape']:.3f}", "Price-to-load inversion quality on the current-day bus profile."),
        ("Forecast Median LMP", f"{future_lmp.mean():.2f}", "Average median future nodal price at the same focus hour."),
    ]
    return "".join(
        f"<div class='stat'><span>{escape(title)}</span><strong>{escape(value)}</strong><p>{escape(detail)}</p></div>"
        for title, value, detail in cards
    )


def _pipeline_strip_html() -> str:
    stages = [
        ("Step 1", "DC-OPF", "Solve the IEEE 14-bus dispatch and recover nodal marginal prices under network constraints."),
        ("Step 2", "Inverse Load Estimation", "Use a surrogate plus inverse optimization to map observed LMPs back to bus demand."),
        ("Step 3", "Sub-Bus Disaggregation", "Allocate parent-bus demand across synthetic lower-voltage sub-buses under conservation constraints."),
        ("Step 4", "Forecast + Price Propagation", "Forecast sub-bus quantiles and push those scenarios through OPF to obtain probabilistic LMP bands."),
    ]
    return "".join(
        f"<div class='stage'><strong>{escape(step)} | {escape(label)}</strong><p>{escape(text)}</p></div>"
        for step, label, text in stages
    )


def _highlights_html(artifacts: WorkflowArtifacts, focus_hour: int) -> str:
    focus_lmp = artifacts.current_opf.bus_lmp.loc[focus_hour]
    focus_load = artifacts.current_truth_bus_load.loc[focus_hour]
    est_load = artifacts.bus_estimate.estimated_bus_load.loc[focus_hour]
    line_snapshot = _line_snapshot_records(artifacts, focus_hour)
    largest_gap_bus = (est_load - focus_load).abs().idxmax()
    largest_gap = float((est_load - focus_load).abs().max())
    highest_price_bus = focus_lmp.idxmax()
    highest_price = float(focus_lmp.max())
    future_uncertainty = (
        artifacts.forecast.bus_lmp_quantiles[max(artifacts.forecast.bus_lmp_quantiles)]
        - artifacts.forecast.bus_lmp_quantiles[min(artifacts.forecast.bus_lmp_quantiles)]
    ).loc[focus_hour]
    widest_bus = future_uncertainty.idxmax()
    widest_band = float(future_uncertainty.max())
    tightest_line = min(line_snapshot, key=lambda row: row["headroom_mw"])
    items = [
        f"<p><strong>{escape(highest_price_bus)}</strong> is the costliest node at hour {focus_hour:02d}, with LMP {highest_price:.2f}.</p>",
        f"<p>The largest one-hour inversion miss appears at <strong>{escape(largest_gap_bus)}</strong>, with absolute load gap {largest_gap:.2f} MW.</p>",
        f"<p>The most stressed line is <strong>{escape(tightest_line['line_name'])}</strong>, carrying {tightest_line['flow_mw']:+.2f} MW with only {tightest_line['headroom_mw']:.2f} MW of remaining margin.</p>",
        f"<p>Future price uncertainty is widest at <strong>{escape(widest_bus)}</strong>, where the P10-P90 spread reaches {widest_band:.2f}.</p>",
        "<p>The market forecast remains physically grounded because future demand scenarios are propagated through repeated OPF solves instead of a direct black-box price model.</p>",
    ]
    return "".join(items)


def _topology_plot_html(artifacts: WorkflowArtifacts, focus_hour: int) -> str:
    graph = nx.Graph()
    graph.add_edges_from(EDGE_LIST)
    focus_lmp = artifacts.current_opf.bus_lmp.loc[focus_hour]
    focus_load = artifacts.current_truth_bus_load.loc[focus_hour]
    estimated_load = artifacts.bus_estimate.estimated_bus_load.loc[focus_hour]
    line_snapshot = _line_snapshot_records(artifacts, focus_hour)
    generator_snapshot = _generator_snapshot_records(artifacts, focus_hour)

    edge_traces = []
    annotations: list[dict[str, Any]] = []
    for row in line_snapshot:
        left = row["from_bus"]
        right = row["to_bus"]
        x0, y0 = BUS_POSITIONS[left]
        x1, y1 = BUS_POSITIONS[right]
        line_color = _loading_color(row["loading_pct"])
        loading_ratio = min(max(row["loading_pct"] / 100.0, 0.0), 1.0)
        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=line_color, width=2.2 + 3.2 * loading_ratio),
                hovertemplate=(
                    f"{row['line_name']}<br>"
                    f"Flow: {row['flow_mw']:+.2f} MW<br>"
                    f"Limit: {row['limit_mw']:.2f} MW<br>"
                    f"Headroom: {row['headroom_mw']:.2f} MW<br>"
                    f"Loading: {row['loading_pct']:.1f}%<extra></extra>"
                ),
            )
        )
        midpoint_x = (x0 + x1) / 2.0
        midpoint_y = (y0 + y1) / 2.0
        annotations.append(
            dict(
                x=midpoint_x,
                y=midpoint_y,
                text=f"{row['flow_mw']:+.0f} MW<br>{row['headroom_mw']:.0f} MW margin",
                showarrow=False,
                font=dict(size=9, color="#5a241a"),
                bgcolor="rgba(255,252,248,0.92)",
                bordercolor=line_color,
                borderpad=2,
            )
        )
        annotations.append(_line_flow_arrow_annotation(x0, y0, x1, y1, row["flow_mw"], line_color))

    nodes = list(graph.nodes())
    node_trace = go.Scatter(
        x=[BUS_POSITIONS[node][0] for node in nodes],
        y=[BUS_POSITIONS[node][1] for node in nodes],
        mode="markers+text",
        text=[node.replace("Bus ", "B") for node in nodes],
        textposition="middle center",
        textfont=dict(color="white", size=11),
        marker=dict(
            size=[17 + float(focus_load.get(node, 0.0)) * 0.32 for node in nodes],
            color=[float(focus_lmp.get(node, 0.0)) for node in nodes],
            colorscale="YlOrRd",
            colorbar=dict(title="Bus LMP"),
            line=dict(width=1.5, color="#1c3248"),
        ),
        hovertext=[
            (
                f"{node}<br>"
                f"Load: {float(focus_load.get(node, 0.0)):.2f} MW<br>"
                f"Estimated Load: {float(estimated_load.get(node, 0.0)):.2f} MW<br>"
                f"LMP: {float(focus_lmp.get(node, 0.0)):.2f}"
            )
            for node in nodes
        ],
        hovertemplate="%{hovertext}<extra></extra>",
    )

    load_bus_nodes = [node for node in nodes if float(focus_load.get(node, 0.0)) > 1e-6]
    for node in load_bus_nodes:
        x, y = BUS_POSITIONS[node]
        annotations.append(
            dict(
                x=x + 0.055,
                y=y + 0.048,
                text=f"{node}<br>LMP {float(focus_lmp[node]):.2f}",
                showarrow=False,
                font=dict(size=10, color="#12324f"),
                bgcolor="rgba(248,252,255,0.94)",
                bordercolor="rgba(18,37,61,0.18)",
                borderpad=3,
            )
        )

    gen_trace = go.Scatter(
        x=[GENERATOR_POSITIONS[row["generator"]][0] for row in generator_snapshot],
        y=[GENERATOR_POSITIONS[row["generator"]][1] for row in generator_snapshot],
        mode="markers+text",
        text=[row["generator"] for row in generator_snapshot],
        textposition="middle center",
        textfont=dict(color="white", size=10),
        marker=dict(size=18, symbol="square", color="#2f6f71", line=dict(width=1.3, color="#17395c")),
        hovertemplate=[
            (
                f"{row['generator']} @ {row['bus']}<br>"
                f"Dispatch: {row['dispatch_mw']:.2f} MW<br>"
                f"Cost: {row['cost']:.2f}<extra></extra>"
            )
            for row in generator_snapshot
        ],
    )
    for row in generator_snapshot:
        x, y = GENERATOR_POSITIONS[row["generator"]]
        annotations.append(
            dict(
                x=x,
                y=y + 0.06,
                text=f"{row['generator']}<br>{row['dispatch_mw']:.1f} MW | c={row['cost']:.1f}",
                showarrow=False,
                font=dict(size=9, color="#124748"),
                bgcolor="rgba(240,251,248,0.94)",
                bordercolor="rgba(47,111,113,0.20)",
                borderpad=3,
            )
        )

    fig = go.Figure(edge_traces + [node_trace, gen_trace])
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=620,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[-0.02, 1.00]),
        yaxis=dict(visible=False, range=[-0.04, 0.82]),
        annotations=annotations,
    )
    fig.add_annotation(
        x=0.995,
        y=0.80,
        xanchor="right",
        yanchor="top",
        align="left",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.90)",
        bordercolor="rgba(18,37,61,0.12)",
        borderpad=6,
        font=dict(size=10, color="#17395c"),
        text=(
            "<b>Line Loading Legend</b><br>"
            "<span style='color:#2e8b57'>Green</span>: loading < 60%<br>"
            "<span style='color:#d4a017'>Amber</span>: 60% - 85%<br>"
            "<span style='color:#c0392b'>Red</span>: > 85%<br>"
            "Arrow shows positive flow direction"
        ),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _generator_snapshot_records(artifacts: WorkflowArtifacts, focus_hour: int) -> list[dict[str, Any]]:
    dispatch = artifacts.current_opf.gen_dispatch.loc[focus_hour]
    costs: pd.Series = artifacts.current_opf.metadata["generator_costs"].loc[focus_hour]
    generator_bus_map: dict[str, str] = artifacts.current_opf.metadata["generator_bus_map"]
    rows = []
    for generator in dispatch.index:
        rows.append(
            {
                "generator": generator,
                "bus": generator_bus_map[generator],
                "dispatch_mw": float(dispatch[generator]),
                "cost": float(costs[generator]),
            }
        )
    return rows


def _line_snapshot_records(artifacts: WorkflowArtifacts, focus_hour: int) -> list[dict[str, Any]]:
    flows = artifacts.current_opf.line_flows.loc[focus_hour]
    line_limits: dict[str, float] = artifacts.current_opf.metadata["line_limits"]
    endpoints: dict[str, dict[str, str]] = artifacts.current_opf.metadata["line_endpoints"]
    rows = []
    for line_name in flows.index:
        limit = float(line_limits[line_name])
        flow = float(flows[line_name])
        headroom = max(limit - abs(flow), 0.0)
        endpoint = endpoints[line_name]
        rows.append(
            {
                "line_name": line_name.replace("Line::", ""),
                "from_bus": endpoint["from_bus"],
                "to_bus": endpoint["to_bus"],
                "flow_mw": flow,
                "limit_mw": limit,
                "headroom_mw": headroom,
                "loading_pct": abs(flow) / max(limit, 1e-6) * 100.0,
            }
        )
    return rows


def _load_comparison_plot_html(truth: pd.DataFrame, estimate: pd.DataFrame, focus_hour: int) -> str:
    selected = ["Bus 2", "Bus 3", "Bus 4", "Bus 5", "Bus 9"]
    palette = ["#17395c", "#2f6f71", "#b24a30", "#c9922e", "#5a4f8c"]
    fig = go.Figure()
    for idx, bus in enumerate(selected):
        color = palette[idx]
        fig.add_trace(go.Scatter(x=truth.index, y=truth[bus], mode="lines", name=f"{bus} truth", line=dict(color=color, width=2)))
        fig.add_trace(
            go.Scatter(
                x=estimate.index,
                y=estimate[bus],
                mode="lines",
                line=dict(color=color, dash="dash", width=2),
                name=f"{bus} estimate",
            )
        )
    fig.add_vline(x=focus_hour, line_width=1.4, line_dash="dot", line_color="#b24a30")
    fig.update_layout(height=420, xaxis_title="Hour", yaxis_title="MW", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0), margin=dict(l=10, r=10, t=10, b=10))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _subbus_quantile_plot_html(quantiles: dict[float, pd.DataFrame]) -> str:
    q10 = quantiles[min(quantiles)]
    q50 = quantiles[sorted(quantiles)[len(quantiles) // 2]]
    q90 = quantiles[max(quantiles)]
    selected = q50.columns[:4]
    palette = ["#2f6f71", "#b24a30", "#c9922e", "#5a4f8c"]
    fig = go.Figure()
    for idx, column in enumerate(selected):
        color = palette[idx]
        fig.add_trace(go.Scatter(x=q50.index, y=q10[column], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=q50.index,
                y=q90[column],
                mode="lines",
                fill="tonexty",
                line=dict(width=0),
                opacity=0.18,
                fillcolor=_rgba_from_hex(color, 0.18),
                name=f"{column} P10-P90",
            )
        )
        fig.add_trace(go.Scatter(x=q50.index, y=q50[column], mode="lines", line=dict(color=color, width=2), name=f"{column} P50"))
    fig.update_layout(height=420, xaxis_title="Hour", yaxis_title="MW", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0), margin=dict(l=10, r=10, t=10, b=10))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _price_quantile_plot_html(quantiles: dict[float, pd.DataFrame], focus_hour: int) -> str:
    q10 = quantiles[min(quantiles)]
    q50 = quantiles[sorted(quantiles)[len(quantiles) // 2]]
    q90 = quantiles[max(quantiles)]
    buses = ["Bus 1", "Bus 4", "Bus 5", "Bus 9", "Bus 14"]
    palette = ["#17395c", "#2f6f71", "#b24a30", "#c9922e", "#5a4f8c"]
    fig = go.Figure()
    for idx, bus in enumerate(buses):
        color = palette[idx]
        fig.add_trace(go.Scatter(x=q10.index, y=q10[bus], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=q90.index,
                y=q90[bus],
                mode="lines",
                fill="tonexty",
                line=dict(width=0),
                opacity=0.18,
                fillcolor=_rgba_from_hex(color, 0.16),
                name=f"{bus} P10-P90",
            )
        )
        fig.add_trace(go.Scatter(x=q50.index, y=q50[bus], mode="lines", line=dict(color=color, width=2), name=f"{bus} P50"))
    fig.add_vline(x=focus_hour, line_width=1.4, line_dash="dot", line_color="#b24a30")
    fig.update_layout(height=420, xaxis_title="Hour", yaxis_title="LMP", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0), margin=dict(l=10, r=10, t=10, b=10))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _bus_snapshot_table_html(artifacts: WorkflowArtifacts, focus_hour: int) -> str:
    focus_lmp = artifacts.current_opf.bus_lmp.loc[focus_hour]
    focus_load = artifacts.current_truth_bus_load.loc[focus_hour]
    est_load = artifacts.bus_estimate.estimated_bus_load.loc[focus_hour]
    rows = []
    for bus in sorted(focus_lmp.index, key=lambda name: int(name.split()[-1])):
        error = float(est_load[bus] - focus_load[bus])
        rows.append(
            f"<tr><td>{escape(bus)}</td><td>{focus_load[bus]:.2f}</td><td>{est_load[bus]:.2f}</td><td>{error:+.2f}</td><td>{focus_lmp[bus]:.2f}</td></tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Bus</th><th>Load (MW)</th><th>Estimated Load</th><th>Gap</th><th>LMP</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _generator_snapshot_table_html(artifacts: WorkflowArtifacts, focus_hour: int) -> str:
    rows = []
    for row in _generator_snapshot_records(artifacts, focus_hour):
        rows.append(
            (
                f"<tr><td>{escape(row['generator'])}</td><td>{escape(row['bus'])}</td>"
                f"<td>{row['dispatch_mw']:.2f}</td><td>{row['cost']:.2f}</td></tr>"
            )
        )
    return (
        "<table>"
        "<thead><tr><th>Generator</th><th>Bus</th><th>Dispatch (MW)</th><th>Cost</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _line_snapshot_table_html(artifacts: WorkflowArtifacts, focus_hour: int) -> str:
    rows = []
    for row in sorted(_line_snapshot_records(artifacts, focus_hour), key=lambda item: item["headroom_mw"]):
        rows.append(
            (
                f"<tr><td>{escape(row['line_name'])}</td><td>{escape(row['from_bus'])}</td><td>{escape(row['to_bus'])}</td>"
                f"<td>{row['flow_mw']:+.2f}</td><td>{row['limit_mw']:.2f}</td><td>{row['headroom_mw']:.2f}</td><td>{row['loading_pct']:.1f}%</td></tr>"
            )
        )
    return (
        "<table>"
        "<thead><tr><th>Line</th><th>From</th><th>To</th><th>Flow (MW)</th><th>Limit</th><th>Margin</th><th>Loading</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _chat_context_payload(artifacts: WorkflowArtifacts, focus_hour: int) -> dict[str, Any]:
    focus_lmp = artifacts.current_opf.bus_lmp.loc[focus_hour]
    focus_load = artifacts.current_truth_bus_load.loc[focus_hour]
    est_load = artifacts.bus_estimate.estimated_bus_load.loc[focus_hour]
    quantiles = artifacts.forecast.bus_lmp_quantiles
    q_lo = quantiles[min(quantiles)].loc[focus_hour]
    q_mid = quantiles[sorted(quantiles)[len(quantiles) // 2]].loc[focus_hour]
    q_hi = quantiles[max(quantiles)].loc[focus_hour]

    bus_snapshot = []
    for bus in sorted(focus_lmp.index, key=lambda name: int(name.split()[-1])):
        bus_snapshot.append(
            {
                "bus": bus,
                "load_mw": round(float(focus_load[bus]), 3),
                "estimated_load_mw": round(float(est_load[bus]), 3),
                "load_gap_mw": round(float(est_load[bus] - focus_load[bus]), 3),
                "lmp": round(float(focus_lmp[bus]), 3),
                "forecast_p10_lmp": round(float(q_lo[bus]), 3),
                "forecast_p50_lmp": round(float(q_mid[bus]), 3),
                "forecast_p90_lmp": round(float(q_hi[bus]), 3),
            }
        )

    top_price_bus = focus_lmp.idxmax()
    top_load_bus = focus_load.idxmax()
    return {
        "case_id": artifacts.config.case_id,
        "focus_hour": focus_hour,
        "run_config": {
            "seed": artifacts.config.seed,
            "training_days": artifacts.config.training_days,
            "demand_scale": artifacts.config.demand_scale,
            "noise_scale": artifacts.config.noise_scale,
            "monte_carlo_samples": artifacts.config.monte_carlo_samples,
        },
        "metrics": {key: round(float(value), 4) for key, value in artifacts.metrics.items()},
        "snapshot_summary": {
            "total_load_mw": round(float(focus_load.sum()), 3),
            "highest_lmp_bus": top_price_bus,
            "highest_lmp": round(float(focus_lmp[top_price_bus]), 3),
            "largest_load_bus": top_load_bus,
            "largest_load_mw": round(float(focus_load[top_load_bus]), 3),
        },
        "bus_snapshot": bus_snapshot,
        "generator_snapshot": [
            {
                "generator": row["generator"],
                "bus": row["bus"],
                "dispatch_mw": round(float(row["dispatch_mw"]), 3),
                "cost": round(float(row["cost"]), 3),
            }
            for row in _generator_snapshot_records(artifacts, focus_hour)
        ],
        "line_snapshot": [
            {
                "line_name": row["line_name"],
                "from_bus": row["from_bus"],
                "to_bus": row["to_bus"],
                "flow_mw": round(float(row["flow_mw"]), 3),
                "limit_mw": round(float(row["limit_mw"]), 3),
                "headroom_mw": round(float(row["headroom_mw"]), 3),
                "loading_pct": round(float(row["loading_pct"]), 3),
            }
            for row in _line_snapshot_records(artifacts, focus_hour)
        ],
    }


def _llm_panel_html(artifacts: WorkflowArtifacts, focus_hour: int) -> str:
    available, status_text = llm_status()
    suggestions = [
        "Summarize the most interesting price pattern at this focus hour.",
        "Which buses look most stressed or congested, and why?",
        "Explain the gap between true and inferred load in plain language.",
        "What does the P10-P90 forecast band suggest about future price risk?",
    ]
    context_json = json.dumps(_chat_context_payload(artifacts, focus_hour), ensure_ascii=True).replace("</", "<\\/")
    suggestion_html = "".join(
        f"<button class='chat-chip' type='button' data-chat-prompt='{escape(prompt, quote=True)}' {'disabled' if not available else ''}>{escape(prompt)}</button>"
        for prompt in suggestions
    )

    disabled_block = ""
    if not available:
        disabled_block = (
            "<div class='chat-message status'>"
            "<strong>Assistant Offline</strong>"
            f"{escape(status_text)} Add the secret to the deployment and reload the page."
            "</div>"
        )

    return f"""
    <div class="chat-shell">
      <div id="llm-transcript" class="chat-transcript">
        <div class="chat-message assistant">
          <strong>Research Assistant</strong>
          Ask about the current run, topology labels, bus-level LMPs, inverse-load errors, or forecast uncertainty.
        </div>
        {disabled_block}
      </div>
      <div class="chat-suggestions">{suggestion_html}</div>
      <div class="chat-controls">
        <textarea id="llm-input" placeholder="Example: Why is Bus 9 relatively expensive at the selected hour?" {'disabled' if not available else ''}></textarea>
        <div class="chat-actions">
          <button id="llm-send" class="chat-send" type="button" {'disabled' if not available else ''}>Ask The Copilot</button>
          <span id="llm-status" class="chat-hint">{escape(status_text)}</span>
        </div>
      </div>
      <script>
        (() => {{
          const enabled = {str(available).lower()};
          const chatContext = {context_json};
          const transcript = document.getElementById("llm-transcript");
          const input = document.getElementById("llm-input");
          const sendButton = document.getElementById("llm-send");
          const statusEl = document.getElementById("llm-status");
          const chips = Array.from(document.querySelectorAll("[data-chat-prompt]"));
          let previousResponseId = null;

          function appendMessage(role, title, text) {{
            const block = document.createElement("div");
            block.className = `chat-message ${{role}}`;
            const safeText = String(text ?? "");
            block.innerHTML = `<strong>${{title}}</strong>${{safeText.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\\n/g, "<br>")}}`;
            transcript.appendChild(block);
            transcript.scrollTop = transcript.scrollHeight;
          }}

          async function sendPrompt(prompt) {{
            if (!enabled) {{
              return;
            }}
            const message = prompt.trim();
            if (!message) {{
              statusEl.textContent = "Enter a question before sending.";
              return;
            }}
            appendMessage("user", "You", message);
            input.value = "";
            sendButton.disabled = true;
            statusEl.textContent = "Thinking about the current run...";
            try {{
              const response = await fetch("/api/chat", {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify({{
                  message,
                  context: chatContext,
                  previous_response_id: previousResponseId,
                }}),
              }});
              const payload = await response.json();
              if (!response.ok) {{
                throw new Error(payload.detail || "The research assistant request failed.");
              }}
              previousResponseId = payload.response_id || previousResponseId;
              appendMessage("assistant", "Research Assistant", payload.reply || "No text response was returned.");
              statusEl.textContent = payload.model ? `Answered with ${{payload.model}}.` : "Answered.";
            }} catch (error) {{
              appendMessage("status", "System", error.message || "The request failed.");
              statusEl.textContent = "The assistant could not answer this request.";
            }} finally {{
              sendButton.disabled = false;
            }}
          }}

          if (enabled) {{
            sendButton.addEventListener("click", () => sendPrompt(input.value));
            input.addEventListener("keydown", (event) => {{
              if (event.key === "Enter" && !event.shiftKey) {{
                event.preventDefault();
                sendPrompt(input.value);
              }}
            }});
            chips.forEach((chip) => {{
              chip.addEventListener("click", () => {{
                input.value = chip.dataset.chatPrompt || "";
                sendPrompt(input.value);
              }});
            }});
          }}
        }})();
      </script>
    </div>
    """


def _rgba_from_hex(color: str, alpha: float) -> str:
    color = color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _loading_color(loading_pct: float) -> str:
    if loading_pct >= 85.0:
        return "#c0392b"
    if loading_pct >= 60.0:
        return "#d4a017"
    return "#2e8b57"


def _line_flow_arrow_annotation(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    flow_mw: float,
    color: str,
) -> dict[str, Any]:
    if flow_mw >= 0:
        start_x, start_y, end_x, end_y = x0, y0, x1, y1
    else:
        start_x, start_y, end_x, end_y = x1, y1, x0, y0

    dx = end_x - start_x
    dy = end_y - start_y
    tail_x = start_x + 0.36 * dx
    tail_y = start_y + 0.36 * dy
    head_x = start_x + 0.58 * dx
    head_y = start_y + 0.58 * dy
    return {
        "x": head_x,
        "y": head_y,
        "ax": tail_x,
        "ay": tail_y,
        "xref": "x",
        "yref": "y",
        "axref": "x",
        "ayref": "y",
        "showarrow": True,
        "arrowhead": 3,
        "arrowsize": 1.1,
        "arrowwidth": 1.8,
        "arrowcolor": color,
        "text": "",
        "opacity": 0.95,
    }


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("lmp_agent.dashboard:app", host=host, port=port, reload=False)
