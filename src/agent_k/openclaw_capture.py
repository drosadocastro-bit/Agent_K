from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_k.openclaw_evaluator import evaluate_trace
from agent_k.openclaw_models import (
    OpenClawCaptureResult,
    OpenClawOutput,
    OpenClawPrompt,
    OpenClawToolUse,
    OpenClawTrace,
)
from agent_k.openclaw_store import append_report, artifact_paths, write_evaluation, write_trace
from agent_k.profiles import AgentProfile


@dataclass
class _SessionState:
    agent_name: str
    prompt: OpenClawPrompt
    metadata: dict[str, Any] = field(default_factory=dict)
    tools: list[OpenClawToolUse] = field(default_factory=list)
    output: OpenClawOutput | None = None
    next_tool_index: int = 1


class OpenClawCapture:
    """Capture OpenClaw hook events, persist trace.json, evaluate, and append reports."""

    def __init__(self, base_dir: str | Path = "reports/openclaw") -> None:
        self.base_dir = Path(base_dir)
        self._sessions: dict[str, _SessionState] = {}

    def before_agent_start(
        self,
        session_id: str,
        *,
        system_instructions: tuple[str, ...] = (),
        user_prompt: str,
        context_items: tuple[str, ...] = (),
        allowed_tools: tuple[str, ...] = (),
        agent_name: str = "openclaw",
        prompt_metadata: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._sessions[session_id] = _SessionState(
            agent_name=agent_name,
            prompt=OpenClawPrompt(
                system_instructions=tuple(system_instructions),
                user_prompt=user_prompt,
                context_items=tuple(context_items),
                allowed_tools=tuple(allowed_tools),
                metadata=dict(prompt_metadata or {}),
            ),
            metadata=dict(metadata or {}),
        )

    def before_tool_call(
        self,
        session_id: str,
        *,
        name: str,
        arguments: dict[str, Any],
        call_id: str | None = None,
        trusted: bool = True,
    ) -> str:
        session = self._require_session(session_id)
        actual_call_id = call_id or f"tool-{session.next_tool_index}-{uuid4().hex[:8]}"
        session.next_tool_index += 1
        session.tools.append(
            OpenClawToolUse(
                call_id=actual_call_id,
                name=name,
                arguments=dict(arguments),
                result=None,
                trusted=trusted,
            )
        )
        return actual_call_id

    def after_tool_call(
        self,
        session_id: str,
        *,
        result: str,
        call_id: str | None = None,
        name: str | None = None,
        trusted: bool | None = None,
    ) -> None:
        session = self._require_session(session_id)
        index = self._find_tool_index(session, call_id=call_id, name=name)
        tool = session.tools[index]
        session.tools[index] = OpenClawToolUse(
            call_id=tool.call_id,
            name=tool.name,
            arguments=tool.arguments,
            result=result,
            trusted=tool.trusted if trusted is None else trusted,
        )

    def agent_end(
        self,
        session_id: str,
        *,
        output: str,
        confidence: float | None = None,
        output_metadata: dict[str, Any] | None = None,
        profile: AgentProfile | None = None,
    ) -> OpenClawCaptureResult:
        session = self._require_session(session_id)
        session.output = OpenClawOutput(
            content=output,
            confidence=confidence,
            metadata=dict(output_metadata or {}),
        )
        trace = OpenClawTrace(
            session_id=session_id,
            agent_name=session.agent_name,
            prompt=session.prompt,
            tools=tuple(session.tools),
            output=session.output,
            metadata=dict(session.metadata),
        )
        trace_path = write_trace(trace, self.base_dir)
        evaluation = evaluate_trace(trace, profile=profile)
        evaluation_json_path, evaluation_markdown_path = write_evaluation(trace, evaluation, self.base_dir)
        report_log_path = append_report(trace, evaluation, self.base_dir)
        del self._sessions[session_id]
        return OpenClawCaptureResult(
            trace=trace,
            evaluation=evaluation,
            artifacts=artifact_paths(
                trace_path,
                evaluation_json_path,
                evaluation_markdown_path,
                report_log_path,
            ),
        )

    def _require_session(self, session_id: str) -> _SessionState:
        if session_id not in self._sessions:
            raise KeyError(f"No OpenClaw session is active for '{session_id}'.")
        return self._sessions[session_id]

    def _find_tool_index(
        self,
        session: _SessionState,
        *,
        call_id: str | None,
        name: str | None,
    ) -> int:
        if call_id is not None:
            for index, tool in enumerate(session.tools):
                if tool.call_id == call_id:
                    return index
            raise KeyError(f"No captured tool call matches call_id '{call_id}'.")

        for index in range(len(session.tools) - 1, -1, -1):
            tool = session.tools[index]
            if tool.result is None and (name is None or tool.name == name):
                return index

        raise KeyError("No pending captured tool call matched the provided event.")
