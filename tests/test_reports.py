from __future__ import annotations

import json

from agent_k.cli import main
from agent_k.report_json import render_json_report
from agent_k.report_markdown import render_markdown_summary
from agent_k.runner import ScenarioRunner
from agent_k.scenarios import sample_scenarios



def test_report_renderers_include_flags_and_caps() -> None:
    report = ScenarioRunner().run(sample_scenarios())

    json_output = render_json_report(report)
    markdown_output = render_markdown_summary(report)

    parsed = json.loads(json_output)
    overconfidence = next(
        result
        for result in parsed["scenario_results"]
        if result["scenario"]["scenario_id"] == "overconfidence_under_weak_evidence"
    )

    assert parsed["tool_name"] == "Agent K"
    assert parsed["flags"]
    assert overconfidence["score"]["max_severity"] == "medium"
    assert overconfidence["score"]["score_cap_applied"] == 0.65
    assert overconfidence["score"]["recommendation"]
    assert overconfidence["score"]["flags"][0]["type"] == "unsupported_certainty"
    assert "## Aggregate Flags" in markdown_output
    assert "`MEDIUM`" in markdown_output
    assert "capped to `0.650`" in markdown_output
    assert "unsupported_certainty" in markdown_output



def test_cli_writes_json_and_markdown_reports(tmp_path) -> None:
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "summary.md"

    exit_code = main(
        [
            "--json-out",
            str(json_path),
            "--markdown-out",
            str(markdown_path),
        ]
    )

    assert exit_code == 0
    assert json_path.exists()
    assert markdown_path.exists()
