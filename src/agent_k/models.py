from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Severity = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    result: str
    trusted: bool = True


@dataclass(frozen=True)
class Assertion:
    subject: str
    predicate: str
    value: str


@dataclass(frozen=True)
class PromptInput:
    system_instructions: tuple[str, ...]
    user_request: str
    context_items: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentOutput:
    content: str
    confidence: float | None
    actions: tuple[str, ...] = ()
    assertions: tuple[Assertion, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScenarioExpectation:
    required_actions: tuple[str, ...] = ()
    forbidden_actions: tuple[str, ...] = ()
    required_markers: tuple[str, ...] = ()
    forbidden_markers: tuple[str, ...] = ()
    scope_forbidden_markers: tuple[str, ...] = ()
    confidence_min: float | None = None
    confidence_max: float | None = None
    require_uncertainty_marker: bool = False
    attack_required_actions: tuple[str, ...] = ()
    attack_forbidden_actions: tuple[str, ...] = ()
    attack_required_markers: tuple[str, ...] = ()
    injection_markers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    name: str
    description: str
    prompt: PromptInput
    output: AgentOutput
    expectation: ScenarioExpectation
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceMetadata:
    scenario_id: str
    scenario_name: str
    scenario_description: str
    tags: tuple[str, ...] = ()
    collector_version: str = "1"


@dataclass(frozen=True)
class Trace:
    prompt: PromptInput
    output: AgentOutput
    metadata: TraceMetadata


@dataclass(frozen=True)
class IntegrityFlag:
    type: str
    severity: Severity
    detail: str


@dataclass(frozen=True)
class ReportFlag:
    scenario_id: str
    scenario_name: str
    type: str
    severity: Severity
    detail: str


@dataclass(frozen=True)
class DimensionScore:
    name: str
    score: float
    rationale: str
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class DimensionEvaluation:
    score: DimensionScore
    flags: tuple[IntegrityFlag, ...] = ()


@dataclass(frozen=True)
class ScenarioScore:
    raw_overall_score: float
    overall_score: float
    verdict: str
    dimensions: dict[str, DimensionScore]
    flags: tuple[IntegrityFlag, ...] = ()
    max_severity: Severity = "low"
    score_cap_applied: float | None = None
    recommendation: str = ""


@dataclass(frozen=True)
class ScenarioResult:
    scenario: Scenario
    trace: Trace
    score: ScenarioScore


@dataclass(frozen=True)
class RunReport:
    tool_name: str
    tool_version: str
    overall_score: float
    verdict: str
    dimension_averages: dict[str, float]
    flags: tuple[ReportFlag, ...] = field(default_factory=tuple)
    scenario_results: tuple[ScenarioResult, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
