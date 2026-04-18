from __future__ import annotations

from agent_k.openclaw_models import OpenClawEvaluation, OpenClawTrace



def render_openclaw_markdown(trace: OpenClawTrace, evaluation: OpenClawEvaluation) -> str:
    lines = [
        "# Agent K OpenClaw Evaluation",
        "",
        f"- Session: `{trace.session_id}`",
        f"- Agent: `{trace.agent_name}`",
        f"- Verdict: **`{evaluation.verdict.upper()}`**",
        f"- Integrity score: `{evaluation.integrity_score:.3f}`",
        f"- Raw score: `{evaluation.raw_score:.3f}`",
        f"- Max severity: `{evaluation.max_severity.upper()}`",
        f"- Recommendation: {evaluation.recommendation}",
    ]

    if evaluation.score_cap_applied is not None:
        lines.append(f"- Score cap applied: `{evaluation.score_cap_applied:.3f}`")
    else:
        lines.append("- Score cap applied: none")

    lines.extend(["", "## Breakdown", ""])

    for name, score in evaluation.breakdown.items():
        lines.append(f"- `{name}`: `{score:.3f}` - {evaluation.details[name]}")

    lines.extend(["", "## Flags", ""])

    if evaluation.flags:
        for flag in evaluation.flags:
            lines.append(f"- `{flag.severity.upper()}` `{flag.type}`: {flag.detail}")
    else:
        lines.append("- No flags raised.")

    return "\n".join(lines)



def render_openclaw_log_entry(trace: OpenClawTrace, evaluation: OpenClawEvaluation) -> str:
    lines = [
        f"## Session `{trace.session_id}`",
        "",
        f"- Agent: `{trace.agent_name}`",
        f"- Verdict: **`{evaluation.verdict.upper()}`**",
        f"- Integrity score: `{evaluation.integrity_score:.3f}`",
        f"- Raw score: `{evaluation.raw_score:.3f}`",
        f"- Max severity: `{evaluation.max_severity.upper()}`",
    ]

    if evaluation.score_cap_applied is not None:
        lines.append(f"- Score cap applied: `{evaluation.score_cap_applied:.3f}`")
    else:
        lines.append("- Score cap applied: none")

    lines.append(f"- Recommendation: {evaluation.recommendation}")

    if evaluation.flags:
        for flag in evaluation.flags:
            lines.append(f"- Flag `{flag.type}` [{flag.severity.upper()}]: {flag.detail}")
    else:
        lines.append("- Flags: none")

    return "\n".join(lines)
