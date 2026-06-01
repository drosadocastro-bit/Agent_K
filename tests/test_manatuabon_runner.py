from __future__ import annotations

from unittest.mock import patch

import pytest

from agent_k.agents.manatuabon import ManatuabonBridgeError
from agent_k.agents.manatuabon import ManatuabonBridgeRunner
from agent_k.openclaw_capture import OpenClawCapture


def test_manatuabon_runner_captures_bridge_answer_and_confidence(tmp_path) -> None:
    capture = OpenClawCapture(base_dir=tmp_path / "openclaw")
    runner = ManatuabonBridgeRunner(capture, host="http://127.0.0.1:7777")

    runner._post_json = lambda path, payload: {
        "answer": "The memory bank says Vela recovered on a short timescale [Memory #7].",
        "sources": [7],
        "confidence": 0.61,
        "confidence_details": {"score": 0.61, "label": "medium"},
    }
    runner._get_json = lambda path: [
        {"id": 7, "summary": "Vela recovery appears rapid in the cited observation."},
    ]

    result = runner.run_prompt("What evidence do we have for Vela recovery?")

    assert result.trace.output.content.startswith("The memory bank says Vela recovered")
    assert result.trace.output.confidence == 0.61
    assert result.trace.prompt.context_items == (
        "Memory #7: Vela recovery appears rapid in the cited observation.",
    )
    assert result.trace.output.metadata["sources"] == [7]
    assert "confidence_calibration" in result.evaluation.breakdown
    assert "citation_grounding" in result.evaluation.dimensions_skipped
    assert not any(flag.type == "hallucinated_citation" for flag in result.evaluation.flags)


def test_manatuabon_runner_handles_missing_sources_without_context(tmp_path) -> None:
    capture = OpenClawCapture(base_dir=tmp_path / "openclaw")
    runner = ManatuabonBridgeRunner(capture, host="http://127.0.0.1:7777")

    runner._post_json = lambda path, payload: {
        "answer": "I do not have enough cited memory support to answer that confidently.",
        "sources": [],
        "confidence": 0.22,
        "confidence_details": {"score": 0.22, "label": "low"},
    }
    runner._get_json = lambda path: []

    result = runner.run_prompt("What is the exact mass of the object?")

    assert result.trace.prompt.context_items == ()
    assert result.trace.output.confidence == 0.22
    assert result.evaluation.max_severity == "low"


def test_manatuabon_runner_wraps_query_timeout_with_actionable_error(tmp_path) -> None:
    capture = OpenClawCapture(base_dir=tmp_path / "openclaw")
    runner = ManatuabonBridgeRunner(capture, host="http://127.0.0.1:7777", timeout=5.0)

    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        with pytest.raises(ManatuabonBridgeError, match="timed out after 5s"):
            runner._post_json("/query", {"prompt": "test"})
