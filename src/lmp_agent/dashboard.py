from __future__ import annotations

import os
from typing import Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from .agent import PricingWorkflowAgent
from .config import RunConfig

app = FastAPI(title="IEEE 14-Bus Pricing Agent Dashboard")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
    <html>
      <head>
        <title>IEEE 14-Bus Pricing Agent</title>
        <style>
          body { font-family: Georgia, serif; margin: 2rem auto; max-width: 960px; color: #112; background: linear-gradient(180deg,#f8f6ef 0%,#eef3fb 100%); }
          h1 { margin-bottom: 0.2rem; }
          .card { background: rgba(255,255,255,0.88); padding: 1.2rem 1.4rem; border-radius: 14px; box-shadow: 0 18px 40px rgba(17,34,68,0.08); margin-bottom: 1rem; }
          a.button { display: inline-block; padding: 0.7rem 1rem; border-radius: 999px; background: #103d5c; color: white; text-decoration: none; }
          code { background: rgba(16,61,92,0.08); padding: 0.15rem 0.35rem; border-radius: 6px; }
        </style>
      </head>
      <body>
        <div class="card">
          <h1>IEEE 14-Bus Pricing Agent</h1>
          <p>A fixed-workflow research dashboard for DC-OPF, bus-load inversion, sub-bus disaggregation, probabilistic load forecasting, and price propagation.</p>
          <p>Run the default IEEE 14-bus scenario or tune the inputs from the query string.</p>
          <p><a class="button" href="/run">Run Default Scenario</a></p>
          <p>Example: <code>/run?seed=11&training_days=10&demand_scale=1.1&noise_scale=0.1</code></p>
          <p>JSON API: <code>/api/run</code></p>
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


@app.get("/run", response_class=HTMLResponse)
def run_page(
    seed: int = 7,
    training_days: int = 16,
    demand_scale: float = 1.0,
    noise_scale: float = 0.08,
    monte_carlo_samples: int = 20,
) -> str:
    config = RunConfig(
        seed=seed,
        training_days=training_days,
        demand_scale=demand_scale,
        noise_scale=noise_scale,
        monte_carlo_samples=monte_carlo_samples,
    )
    artifacts = PricingWorkflowAgent(config).run_pipeline()

    network_plot = _network_plot_html(artifacts.current_opf.bus_lmp.mean())
    load_plot = _load_comparison_plot_html(
        artifacts.current_truth_bus_load, artifacts.bus_estimate.estimated_bus_load
    )
    subbus_plot = _subbus_quantile_plot_html(artifacts.forecast.subbus_load_quantiles)
    price_plot = _price_quantile_plot_html(artifacts.forecast.bus_lmp_quantiles)
    metric_items = "".join(
        f"<li><strong>{key}</strong>: {value:.4f}</li>"
        for key, value in artifacts.metrics.items()
    )

    return f"""
    <html>
      <head>
        <title>Run Result</title>
        <style>
          body {{ font-family: Georgia, serif; margin: 2rem auto; max-width: 1180px; color: #112; background: linear-gradient(180deg,#f8f6ef 0%,#eef3fb 100%); }}
          .hero, .panel {{ background: rgba(255,255,255,0.90); border-radius: 16px; padding: 1.2rem 1.4rem; box-shadow: 0 18px 40px rgba(17,34,68,0.08); margin-bottom: 1rem; }}
          .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
          h1, h2 {{ margin-top: 0; }}
          a.button {{ display: inline-block; padding: 0.7rem 1rem; border-radius: 999px; background: #103d5c; color: white; text-decoration: none; }}
          ul {{ line-height: 1.7; }}
          pre {{ white-space: pre-wrap; }}
        </style>
      </head>
      <body>
        <div class="hero">
          <h1>Run Summary</h1>
          <p>seed={seed}, training_days={training_days}, demand_scale={demand_scale}, noise_scale={noise_scale}, monte_carlo_samples={monte_carlo_samples}</p>
          <p><a class="button" href="/">Back</a></p>
          <ul>{metric_items}</ul>
        </div>
        <div class="grid">
          <div class="panel"><h2>Average Bus LMP Heat Graph</h2>{network_plot}</div>
          <div class="panel"><h2>Bus Truth vs Estimated Load</h2>{load_plot}</div>
          <div class="panel"><h2>Sub-bus Forecast Quantiles</h2>{subbus_plot}</div>
          <div class="panel"><h2>Bus LMP Quantile Forecast</h2>{price_plot}</div>
        </div>
        <div class="panel">
          <h2>Report</h2>
          <pre>{artifacts.report_markdown}</pre>
        </div>
      </body>
    </html>
    """


def _network_plot_html(mean_lmp: pd.Series) -> str:
    graph = nx.Graph()
    graph.add_edges_from(
        [
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
    )
    positions = nx.spring_layout(graph, seed=14)
    traces = []
    for left, right in graph.edges():
        x0, y0 = positions[left]
        x1, y1 = positions[right]
        traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="#96a6b7", width=1.5),
                hoverinfo="skip",
            )
        )
    traces.append(
        go.Scatter(
            x=[positions[node][0] for node in graph.nodes()],
            y=[positions[node][1] for node in graph.nodes()],
            mode="markers+text",
            text=list(graph.nodes()),
            textposition="top center",
            marker=dict(
                size=20,
                color=[float(mean_lmp.get(node, 0.0)) for node in graph.nodes()],
                colorscale="YlOrRd",
                colorbar=dict(title="LMP"),
                line=dict(width=1, color="#21364d"),
            ),
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig = go.Figure(traces)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=420)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _load_comparison_plot_html(truth: pd.DataFrame, estimate: pd.DataFrame) -> str:
    selected = ["Bus 2", "Bus 3", "Bus 4", "Bus 5", "Bus 9"]
    fig = go.Figure()
    for bus in selected:
        fig.add_trace(go.Scatter(x=truth.index, y=truth[bus], mode="lines", name=f"{bus} truth"))
        fig.add_trace(
            go.Scatter(
                x=estimate.index,
                y=estimate[bus],
                mode="lines",
                line=dict(dash="dash"),
                name=f"{bus} estimate",
            )
        )
    fig.update_layout(height=420, xaxis_title="Hour", yaxis_title="MW", legend=dict(orientation="h"))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _subbus_quantile_plot_html(quantiles: dict[float, pd.DataFrame]) -> str:
    q10 = quantiles[min(quantiles)]
    q50 = quantiles[sorted(quantiles)[len(quantiles) // 2]]
    q90 = quantiles[max(quantiles)]
    selected = q50.columns[:4]
    fig = go.Figure()
    for column in selected:
        fig.add_trace(go.Scatter(x=q50.index, y=q10[column], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=q50.index,
                y=q90[column],
                mode="lines",
                fill="tonexty",
                line=dict(width=0),
                opacity=0.22,
                name=f"{column} P10-P90",
            )
        )
        fig.add_trace(go.Scatter(x=q50.index, y=q50[column], mode="lines", name=f"{column} P50"))
    fig.update_layout(height=420, xaxis_title="Hour", yaxis_title="MW", legend=dict(orientation="h"))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _price_quantile_plot_html(quantiles: dict[float, pd.DataFrame]) -> str:
    q10 = quantiles[min(quantiles)]
    q50 = quantiles[sorted(quantiles)[len(quantiles) // 2]]
    q90 = quantiles[max(quantiles)]
    buses = ["Bus 1", "Bus 4", "Bus 5", "Bus 9", "Bus 14"]
    fig = go.Figure()
    for bus in buses:
        fig.add_trace(go.Scatter(x=q10.index, y=q10[bus], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=q90.index,
                y=q90[bus],
                mode="lines",
                fill="tonexty",
                line=dict(width=0),
                opacity=0.20,
                name=f"{bus} P10-P90",
            )
        )
        fig.add_trace(go.Scatter(x=q50.index, y=q50[bus], mode="lines", name=f"{bus} P50"))
    fig.update_layout(height=420, xaxis_title="Hour", yaxis_title="LMP", legend=dict(orientation="h"))
    return fig.to_html(full_html=False, include_plotlyjs=False)


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("lmp_agent.dashboard:app", host=host, port=port, reload=False)
