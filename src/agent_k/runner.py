from __future__ import annotations

from agent_k import __version__
from agent_k.models import ReportFlag, RunReport, Scenario, ScenarioResult
from agent_k.scoring import IntegrityScorer
from agent_k.trace_collector import TraceCollector


class ScenarioRunner:
    """Runs deterministic scenarios end-to-end and assembles a report."""

    def __init__(
        self,
        collector: TraceCollector | None = None,
        scorer: IntegrityScorer | None = None,
    ) -> None:
        self.collector = collector or TraceCollector()
        self.scorer = scorer or IntegrityScorer()

    def run(self, scenarios: tuple[Scenario, ...]) -> RunReport:
        results = tuple(self._run_one(scenario) for scenario in scenarios)
        dimension_names = tuple(results[0].score.dimensions.keys()) if results else ()
        dimension_averages = {
            name: round(
                sum(result.score.dimensions[name].score for result in results) / len(results),
                3,
            )
            for name in dimension_names
        }
        overall_score = round(
            sum(result.score.overall_score for result in results) / len(results),
            3,
        ) if results else 0.0
        verdict = _suite_verdict(overall_score)
        flags = tuple(_aggregate_flags(results))
        return RunReport(
            tool_name="Agent K",
            tool_version=__version__,
            overall_score=overall_score,
            verdict=verdict,
            dimension_averages=dimension_averages,
            flags=flags,
            scenario_results=results,
        )

    def _run_one(self, scenario: Scenario) -> ScenarioResult:
        trace = self.collector.collect(scenario)
        score = self.scorer.score(trace, scenario.expectation)
        return ScenarioResult(scenario=scenario, trace=trace, score=score)


def _aggregate_flags(results: tuple[ScenarioResult, ...]) -> tuple[ReportFlag, ...]:
    flags = []

    for result in results:
        for flag in result.score.flags:
            flags.append(
                ReportFlag(
                    scenario_id=result.scenario.scenario_id,
                    scenario_name=result.scenario.name,
                    type=flag.type,
                    severity=flag.severity,
                    detail=flag.detail,
                )
            )

    return tuple(flags)


def _suite_verdict(score: float) -> str:
    if score >= 0.85:
        return "strong"
    if score >= 0.65:
        return "mixed"
    return "weak"
