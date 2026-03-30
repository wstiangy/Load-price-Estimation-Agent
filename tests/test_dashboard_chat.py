import pytest
from fastapi import HTTPException

from lmp_agent.agent import PricingWorkflowAgent
from lmp_agent.config import RunConfig
from lmp_agent.dashboard import ChatRequest, _chat_context_payload, _llm_panel_html, api_chat, run_page


def test_chat_context_includes_all_buses():
    artifacts = PricingWorkflowAgent(RunConfig(training_days=6, monte_carlo_samples=4, seed=3)).run_pipeline()
    context = _chat_context_payload(artifacts, focus_hour=12)
    assert context["case_id"] == "ieee14"
    assert context["focus_hour"] == 12
    assert len(context["bus_snapshot"]) == 14
    assert len(context["generator_snapshot"]) == 5
    assert len(context["line_snapshot"]) == 20
    assert context["snapshot_summary"]["highest_lmp_bus"].startswith("Bus ")


def test_llm_panel_shows_disabled_message_when_key_missing(monkeypatch):
    artifacts = PricingWorkflowAgent(RunConfig(training_days=6, monte_carlo_samples=4, seed=4)).run_pipeline()
    monkeypatch.setattr("lmp_agent.dashboard.llm_status", lambda: (False, "Set OPENAI_API_KEY to enable chat."))
    html = _llm_panel_html(artifacts, focus_hour=18)
    assert "Assistant Offline" in html
    assert "OPENAI_API_KEY" in html


def test_api_chat_maps_runtime_error_to_service_unavailable(monkeypatch):
    monkeypatch.setattr(
        "lmp_agent.dashboard.generate_chat_reply",
        lambda **_: (_ for _ in ()).throw(RuntimeError("OPENAI_API_KEY is not configured.")),
    )
    request = ChatRequest(message="Explain Bus 9.", context={"case_id": "ieee14"})
    with pytest.raises(HTTPException) as exc_info:
        api_chat(request)
    assert exc_info.value.status_code == 503
    assert "OPENAI_API_KEY" in str(exc_info.value.detail)


def test_run_page_contains_generator_and_line_sections():
    html = run_page(training_days=6, monte_carlo_samples=4, focus_hour=18)
    assert "Generator Dispatch and Marginal Cost" in html
    assert "Line Loading and Congestion Margin" in html
    assert "Single-Line Topology" in html
