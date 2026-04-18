from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agent_k.models import IntegrityFlag, Severity


@dataclass(frozen=True)
class OpenClawPrompt:
    system_instructions: tuple[str, ...] = ()
    user_prompt: str = ""
    context_items: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenClawToolUse:
    call_id: str
    name: str
    arguments: dict[str, Any]
    result: str | None = None
    trusted: bool = True


@dataclass(frozen=True)
class OpenClawOutput:
    content: str
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenClawTrace:
    session_id: str
    agent_name: str = "openclaw"
    prompt: OpenClawPrompt = field(default_factory=OpenClawPrompt)
    tools: tuple[OpenClawToolUse, ...] = ()
    output: OpenClawOutput | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OpenClawEvaluation:
    session_id: str
    integrity_score: float
    raw_score: float
    breakdown: dict[str, float]
    details: dict[str, str]
    flags: tuple[IntegrityFlag, ...] = ()
    max_severity: Severity = "low"
    score_cap_applied: float | None = None
    recommendation: str = ""
    dimensions_skipped: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OpenClawCaptureArtifact:
    trace_path: str
    evaluation_json_path: str
    evaluation_markdown_path: str
    report_log_path: str


@dataclass(frozen=True)
class OpenClawCaptureResult:
    trace: OpenClawTrace
    evaluation: OpenClawEvaluation
    artifacts: OpenClawCaptureArtifact


def trace_from_dict(data: dict[str, Any]) -> OpenClawTrace:
    prompt_data = data.get("prompt", {})
    tool_items = tuple(
        OpenClawToolUse(
            call_id=item["call_id"],
            name=item["name"],
            arguments=dict(item.get("arguments", {})),
            result=item.get("result"),
            trusted=item.get("trusted", True),
        )
        for item in data.get("tools", ())
    )
    output_data = data.get("output")
    output = None
    if output_data is not None:
        output = OpenClawOutput(
            content=output_data.get("content", ""),
            confidence=output_data.get("confidence"),
            metadata=dict(output_data.get("metadata", {})),
        )

    return OpenClawTrace(
        session_id=data["session_id"],
        agent_name=data.get("agent_name", "openclaw"),
        prompt=OpenClawPrompt(
            system_instructions=tuple(prompt_data.get("system_instructions", ())),
            user_prompt=prompt_data.get("user_prompt", ""),
            context_items=tuple(prompt_data.get("context_items", ())),
            allowed_tools=tuple(prompt_data.get("allowed_tools", ())),
            metadata=dict(prompt_data.get("metadata", {})),
        ),
        tools=tool_items,
        output=output,
        metadata=dict(data.get("metadata", {})),
    )
