from __future__ import annotations

from agent_k.runner import ScenarioRunner
from agent_k.scenarios import sample_scenarios


def test_runner_is_deterministic() -> None:
    runner = ScenarioRunner()

    first = runner.run(sample_scenarios()).to_dict()
    second = runner.run(sample_scenarios()).to_dict()

    assert first == second


def test_all_sample_scenarios_are_present() -> None:
    report = ScenarioRunner().run(sample_scenarios())
    scenario_ids = [result.scenario.scenario_id for result in report.scenario_results]

    assert scenario_ids == [
        "normal_task",
        "conflicting_instructions",
        "prompt_injection_attempt",
        "overconfidence_under_weak_evidence",
        "sycophancy_under_pushback",
        "hallucinated_citation_attempt",
    ]


def test_overconfidence_scenario_scores_lower_than_baseline() -> None:
    report = ScenarioRunner().run(sample_scenarios())
    results = {result.scenario.scenario_id: result for result in report.scenario_results}

    assert (
        results["normal_task"].score.overall_score
        > results["overconfidence_under_weak_evidence"].score.overall_score
    )
