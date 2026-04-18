from __future__ import annotations

from agent_k.models import RunReport, ScenarioResult


def render_markdown_summary(report: RunReport) -> str:
    lines = [
        "# Agent K v1 Integrity Summary",
        "",
        f"- Tool version: `{report.tool_version}`",
        f"- Overall score: `{report.overall_score:.3f}`",
        f"- Verdict: `{report.verdict}`",
        "",
        "## Dimension Averages",
        "",
        "| Dimension | Score |",
        "| --- | ---: |",
    ]

    for name, score in report.dimension_averages.items():
        lines.append(f"| `{name}` | `{score:.3f}` |")

    lines.extend(
        [
            "",
            "## Aggregate Flags",
            "",
        ]
    )

    if report.flags:
        for flag in report.flags:
            lines.append(
                f"- `{flag.severity.upper()}` `{flag.type}` in `{flag.scenario_id}`: {flag.detail}"
            )
    else:
        lines.append("- No scenario flags were raised.")

    lines.extend(
        [
            "",
            "## Scenario Overview",
            "",
            "| Scenario | Final | Raw | Verdict | Severity | Cap |",
            "| --- | ---: | ---: | --- | --- | --- |",
        ]
    )

    for result in report.scenario_results:
        cap_text = (
            f"`{result.score.score_cap_applied:.3f}`"
            if result.score.score_cap_applied is not None
            else "`none`"
        )
        lines.append(
            f"| `{result.scenario.scenario_id}` | `{result.score.overall_score:.3f}` | "
            f"`{result.score.raw_overall_score:.3f}` | `{result.score.verdict}` | "
            f"`{result.score.max_severity.upper()}` | {cap_text} |"
        )

    lines.extend(["", "## Scenario Details", ""])

    for result in report.scenario_results:
        lines.extend(_scenario_detail_lines(result))

    return "\n".join(lines)


def _scenario_detail_lines(result: ScenarioResult) -> list[str]:
    score = result.score
    lines = [
        f"### `{result.scenario.scenario_id}`",
        "",
        f"- Severity: `{score.max_severity.upper()}`",
        f"- Verdict: `{score.verdict}`",
        f"- Raw score: `{score.raw_overall_score:.3f}`",
        f"- Final score: `{score.overall_score:.3f}`",
    ]

    if score.score_cap_applied is not None:
        lines.append(f"- Cap applied: capped to `{score.score_cap_applied:.3f}`")
    else:
        lines.append("- Cap applied: none")

    lines.append(f"- Explanation: {score.recommendation}")

    if score.flags:
        for flag in score.flags:
            lines.append(
                f"- Flag `{flag.type}` [{flag.severity.upper()}]: {flag.detail}"
            )
    else:
        lines.append("- Flags: none")

    lines.append("")
    return lines
