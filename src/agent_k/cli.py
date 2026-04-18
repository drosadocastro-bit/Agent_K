from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_k.report_json import render_json_report
from agent_k.report_markdown import render_markdown_summary
from agent_k.runner import ScenarioRunner
from agent_k.scenarios import sample_scenarios, scenario_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Agent K deterministic integrity scenarios.")
    parser.add_argument(
        "--scenario",
        action="append",
        choices=("all", *scenario_ids()),
        help="Scenario id to run. Defaults to all built-in scenarios.",
    )
    parser.add_argument(
        "--json-out",
        default="reports/agent_k_report.json",
        help="Path for the JSON report output.",
    )
    parser.add_argument(
        "--markdown-out",
        default="reports/agent_k_summary.md",
        help="Path for the Markdown summary output.",
    )
    return parser


def _build_live_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-k live",
        description=(
            "Run the built-in scenarios against a local LLM (opt-in, network to "
            "configured host only). Default 'python -m agent_k' stays fully offline."
        ),
    )
    parser.add_argument("--provider", choices=("ollama",), default="ollama")
    parser.add_argument("--model", required=True, help="Model identifier for the provider (e.g. qwen3:4b).")
    parser.add_argument(
        "--scenario",
        action="append",
        choices=("all", *scenario_ids()),
        help="Scenario id to run. Defaults to all built-in scenarios.",
    )
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-dir",
        default="reports/openclaw",
        help="Directory for OpenClaw traces and per-session reports.",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Optional path to write a cross-scenario JSON summary for this run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Dispatch: the first positional "live" selects the live-agent subcommand.
    # Everything else preserves the legacy offline CLI contract.
    import sys

    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    if effective_argv and effective_argv[0] == "live":
        return _run_live(effective_argv[1:])

    parser = build_parser()
    args = parser.parse_args(effective_argv)
    selected = _select_scenarios(tuple(args.scenario or ("all",)))
    report = ScenarioRunner().run(selected)

    json_output = render_json_report(report)
    markdown_output = render_markdown_summary(report)

    _write_output(Path(args.json_out), json_output)
    _write_output(Path(args.markdown_out), markdown_output)

    print(markdown_output)
    return 0


def _run_live(argv: list[str]) -> int:
    # Local import keeps the default offline path free of any optional imports.
    from agent_k.agents.ollama import OllamaRunner
    from agent_k.openclaw_capture import OpenClawCapture

    args = _build_live_parser().parse_args(argv)
    scenarios = _select_scenarios(tuple(args.scenario or ("all",)))
    capture = OpenClawCapture(base_dir=args.base_dir)
    runner = OllamaRunner(capture, model=args.model, host=args.host, seed=args.seed)

    summary: list[dict] = []
    print(f"Running {len(scenarios)} scenario(s) against {args.provider}:{args.model}")
    for scenario in scenarios:
        session_id = f"{args.provider}-{args.model.replace(':', '-')}-{scenario.scenario_id}"
        result = runner.run_scenario(scenario, session_id=session_id)
        evaluation = result.evaluation
        flag_types = sorted({flag.type for flag in evaluation.flags})
        summary.append(
            {
                "scenario_id": scenario.scenario_id,
                "session_id": session_id,
                "integrity_score": evaluation.integrity_score,
                "max_severity": evaluation.max_severity,
                "score_cap_applied": evaluation.score_cap_applied,
                "flag_types": flag_types,
                "breakdown": evaluation.breakdown,
                "recommendation": evaluation.recommendation,
                "output_preview": result.trace.output.content[:200],
            }
        )
        cap = evaluation.score_cap_applied
        cap_text = f"cap={cap}" if cap is not None else "cap=none"
        print(
            f"  {scenario.scenario_id:<38} "
            f"score={evaluation.integrity_score:.3f} "
            f"severity={evaluation.max_severity:<6} "
            f"{cap_text:<10} "
            f"flags={','.join(flag_types) or 'none'}"
        )

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {"provider": args.provider, "model": args.model, "results": summary},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"Wrote summary: {summary_path}")

    return 0


def _select_scenarios(selected_ids: tuple[str, ...]):
    scenarios = sample_scenarios()
    if "all" in selected_ids:
        return scenarios

    selected_lookup = set(selected_ids)
    return tuple(scenario for scenario in scenarios if scenario.scenario_id in selected_lookup)


def _write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
