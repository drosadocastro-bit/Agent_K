from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from agent_k.openclaw_evaluator import evaluate_trace
from agent_k.openclaw_models import trace_from_dict
from agent_k.openclaw_report_json import render_openclaw_json
from agent_k.openclaw_report_markdown import render_openclaw_markdown
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
    parser.add_argument(
        "--allow-remote-host",
        action="store_true",
        help="Allow sending scenario prompts to a non-local host. Defaults to localhost-only.",
    )
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


def _build_bridge_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-k bridge",
        description="Query a running Manatuabon bridge and score the response with Agent K.",
    )
    parser.add_argument("prompt", help="Prompt to send to the bridge.")
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:7777",
        help="Base URL of the Manatuabon bridge.",
    )
    parser.add_argument(
        "--allow-remote-host",
        action="store_true",
        help="Allow sending bridge prompts to a non-local host. Defaults to localhost-only.",
    )
    parser.add_argument(
        "--base-dir",
        default="reports/openclaw",
        help="Directory for OpenClaw traces and per-session reports.",
    )
    parser.add_argument(
        "--session-id",
        default="manatuabon-bridge-eval",
        help="Session id for the captured run.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds for bridge requests.",
    )
    return parser


def _build_eval_trace_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-k eval-trace",
        description="Evaluate an existing OpenClaw trace JSON file with Agent K.",
    )
    parser.add_argument("trace_path", help="Path to an OpenClaw trace JSON file.")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Path for the evaluation JSON output. Defaults to evaluation.json next to the trace.",
    )
    parser.add_argument(
        "--markdown-out",
        default=None,
        help="Path for the evaluation Markdown output. Defaults to evaluation.md next to the trace.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Dispatch: the first positional "live" selects the live-agent subcommand.
    # Everything else preserves the legacy offline CLI contract.
    import sys

    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    if effective_argv and effective_argv[0] == "live":
        return _run_live(effective_argv[1:])
    if effective_argv and effective_argv[0] == "bridge":
        return _run_bridge(effective_argv[1:])
    if effective_argv and effective_argv[0] == "eval-trace":
        return _run_eval_trace(effective_argv[1:])

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
    if not _validate_local_host(args.host, allow_remote_host=args.allow_remote_host):
        return 2
    scenarios = _select_scenarios(tuple(args.scenario or ("all",)))
    capture = OpenClawCapture(base_dir=args.base_dir)
    runner = OllamaRunner(
        capture,
        model=args.model,
        host=args.host,
        seed=args.seed,
        allow_remote_host=args.allow_remote_host,
    )

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
                "verdict": evaluation.verdict,
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
            f"verdict={evaluation.verdict:<9} "
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


def _run_bridge(argv: list[str]) -> int:
    from agent_k.agents.manatuabon import ManatuabonBridgeRunner
    from agent_k.agents.manatuabon import ManatuabonBridgeError
    from agent_k.openclaw_capture import OpenClawCapture
    import sys

    args = _build_bridge_parser().parse_args(argv)
    if not _validate_local_host(args.host, allow_remote_host=args.allow_remote_host):
        return 2
    capture = OpenClawCapture(base_dir=args.base_dir)
    runner = ManatuabonBridgeRunner(
        capture,
        host=args.host,
        timeout=args.timeout,
        allow_remote_host=args.allow_remote_host,
    )
    try:
        result = runner.run_prompt(args.prompt, session_id=args.session_id)
    except ManatuabonBridgeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    evaluation = result.evaluation
    flag_types = sorted({flag.type for flag in evaluation.flags})

    print(
        f"verdict={evaluation.verdict:<9} "
        f"score={evaluation.integrity_score:.3f} "
        f"severity={evaluation.max_severity:<6} "
        f"flags={','.join(flag_types) or 'none'}"
    )
    print(f"trace={result.artifacts.trace_path}")
    print(f"evaluation_json={result.artifacts.evaluation_json_path}")
    print(f"evaluation_markdown={result.artifacts.evaluation_markdown_path}")
    print(f"report_log={result.artifacts.report_log_path}")
    return 0


def _run_eval_trace(argv: list[str]) -> int:
    import sys

    args = _build_eval_trace_parser().parse_args(argv)
    trace_path = Path(args.trace_path)
    json_path = Path(args.json_out) if args.json_out else trace_path.parent / "evaluation.json"
    markdown_path = Path(args.markdown_out) if args.markdown_out else trace_path.parent / "evaluation.md"

    try:
        trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
        trace = trace_from_dict(trace_data)
        evaluation = evaluate_trace(trace)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"Could not evaluate trace '{trace_path}': {exc}", file=sys.stderr)
        return 2

    _write_output(json_path, render_openclaw_json(evaluation))
    _write_output(markdown_path, render_openclaw_markdown(trace, evaluation))

    flag_types = sorted({flag.type for flag in evaluation.flags})
    print(
        f"verdict={evaluation.verdict:<9} "
        f"score={evaluation.integrity_score:.3f} "
        f"severity={evaluation.max_severity:<6} "
        f"flags={','.join(flag_types) or 'none'}"
    )
    print(f"evaluation_json={json_path}")
    print(f"evaluation_markdown={markdown_path}")
    return 0


def _select_scenarios(selected_ids: tuple[str, ...]):
    scenarios = sample_scenarios()
    if "all" in selected_ids:
        return scenarios

    selected_lookup = set(selected_ids)
    return tuple(scenario for scenario in scenarios if scenario.scenario_id in selected_lookup)


def _validate_local_host(host: str, *, allow_remote_host: bool) -> bool:
    if allow_remote_host:
        return True

    import sys

    parsed = urlparse(host)
    hostname = parsed.hostname
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    if hostname in local_hosts:
        return True

    print(
        f"Refusing non-local host '{host}'. Re-run with --allow-remote-host to acknowledge network egress.",
        file=sys.stderr,
    )
    return False


def _write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
