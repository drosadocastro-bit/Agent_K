from __future__ import annotations

from pathlib import Path

from agent_k.openclaw_models import (
    OpenClawCaptureArtifact,
    OpenClawEvaluation,
    OpenClawTrace,
)
from agent_k.openclaw_report_json import render_openclaw_json
from agent_k.openclaw_report_markdown import (
    render_openclaw_log_entry,
    render_openclaw_markdown,
)



def write_trace(trace: OpenClawTrace, base_dir: Path) -> Path:
    session_dir = _session_dir(base_dir, trace.session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    trace_path = session_dir / "trace.json"
    trace_path.write_text(_json_dump(trace.to_dict()), encoding="utf-8")
    return trace_path



def write_evaluation(
    trace: OpenClawTrace,
    evaluation: OpenClawEvaluation,
    base_dir: Path,
) -> tuple[Path, Path]:
    session_dir = _session_dir(base_dir, trace.session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    evaluation_json_path = session_dir / "evaluation.json"
    evaluation_markdown_path = session_dir / "evaluation.md"
    evaluation_json_path.write_text(render_openclaw_json(evaluation), encoding="utf-8")
    evaluation_markdown_path.write_text(
        render_openclaw_markdown(trace, evaluation),
        encoding="utf-8",
    )
    return evaluation_json_path, evaluation_markdown_path



def append_report(trace: OpenClawTrace, evaluation: OpenClawEvaluation, base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    report_log_path = base_dir / "report_log.md"
    entry = render_openclaw_log_entry(trace, evaluation)
    existing = report_log_path.read_text(encoding="utf-8") if report_log_path.exists() else ""
    separator = "\n\n" if existing.strip() else ""
    report_log_path.write_text(existing + separator + entry, encoding="utf-8")
    return report_log_path



def artifact_paths(
    trace_path: Path,
    evaluation_json_path: Path,
    evaluation_markdown_path: Path,
    report_log_path: Path,
) -> OpenClawCaptureArtifact:
    return OpenClawCaptureArtifact(
        trace_path=str(trace_path),
        evaluation_json_path=str(evaluation_json_path),
        evaluation_markdown_path=str(evaluation_markdown_path),
        report_log_path=str(report_log_path),
    )



def _session_dir(base_dir: Path, session_id: str) -> Path:
    return base_dir / session_id



def _json_dump(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2)
