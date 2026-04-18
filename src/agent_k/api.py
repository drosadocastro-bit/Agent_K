from __future__ import annotations

from typing import Any

from agent_k.openclaw_evaluator import evaluate_trace
from agent_k.openclaw_models import OpenClawEvaluation, OpenClawTrace
from agent_k.profiles import AgentProfile



def evaluate(
    trace: OpenClawTrace | dict[str, Any],
    *,
    profile: AgentProfile | None = None,
) -> OpenClawEvaluation:
    return evaluate_trace(trace, profile=profile)
