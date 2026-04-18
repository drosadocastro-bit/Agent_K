"""Provider-agnostic base class for live agent runners.

A runner takes a built-in Agent K :class:`Scenario`, drives a real LLM through
the scenario's prompt, and hands the captured session to ``OpenClawCapture``
for deterministic evaluation. The runner never scores anything itself — that
is intentional, so scoring stays offline, reproducible, and auditable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_k.openclaw_capture import OpenClawCapture
from agent_k.openclaw_models import OpenClawCaptureResult
from agent_k.models import Scenario
from agent_k.profiles import UNCALIBRATED_CHAT, AgentProfile


class AgentRunner(ABC):
    """Abstract base class for provider-specific live agent runners."""

    provider: str = "unknown"

    #: Declared capability profile for this runner. Base chat LLMs default
    #: to ``UNCALIBRATED_CHAT`` because they do not emit calibrated numeric
    #: confidence; subclasses for structured-output agents can override.
    profile: AgentProfile = UNCALIBRATED_CHAT

    def __init__(self, capture: OpenClawCapture) -> None:
        self.capture = capture

    @abstractmethod
    def complete(
        self,
        *,
        system_instructions: tuple[str, ...],
        user_prompt: str,
        context_items: tuple[str, ...],
    ) -> tuple[str, dict[str, Any]]:
        """Send the prompt to the underlying LLM and return ``(content, metadata)``.

        ``metadata`` is recorded on the captured output so reviewers can trace
        back exactly which model/digest/parameters produced the response.
        Implementations must not raise on ordinary empty responses; they
        should return ``("", metadata)`` instead so the evaluator can still
        score the absence of content.
        """

    def run_scenario(self, scenario: Scenario, *, session_id: str | None = None) -> OpenClawCaptureResult:
        """Drive ``scenario`` through the provider and return the captured result."""
        effective_session_id = session_id or f"{self.provider}-{scenario.scenario_id}"

        self.capture.before_agent_start(
            effective_session_id,
            system_instructions=scenario.prompt.system_instructions,
            user_prompt=scenario.prompt.user_request,
            context_items=scenario.prompt.context_items,
            allowed_tools=scenario.prompt.allowed_tools,
            agent_name=f"{self.provider}:{getattr(self, 'model', 'unknown')}",
            prompt_metadata={"scenario_id": scenario.scenario_id},
        )

        content, output_metadata = self.complete(
            system_instructions=scenario.prompt.system_instructions,
            user_prompt=scenario.prompt.user_request,
            context_items=scenario.prompt.context_items,
        )

        # Confidence is intentionally None: these base-chat models do not
        # produce calibrated numeric confidence. The evaluator honours the
        # runner's declared ``profile`` to decide whether that absence is a
        # finding (default profile) or a known-N/A dimension (uncalibrated
        # profile) — heuristics and scoring math are unchanged either way.
        return self.capture.agent_end(
            effective_session_id,
            output=content,
            confidence=None,
            output_metadata=output_metadata,
            profile=self.profile,
        )
