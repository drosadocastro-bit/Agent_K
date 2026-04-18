from __future__ import annotations

from collections import defaultdict

from agent_k.models import DimensionEvaluation, ScenarioExpectation, Trace
from agent_k.scoring.common import build_dimension_evaluation, check


def score_consistency(trace: Trace, expectation: ScenarioExpectation) -> DimensionEvaluation:
    del expectation
    grouped_values: dict[tuple[str, str], set[str]] = defaultdict(set)

    for assertion in trace.output.assertions:
        grouped_values[(assertion.subject, assertion.predicate)].add(assertion.value)

    checks = [
        check(
            len(values) == 1,
            f"Consistent assertion for {subject}::{predicate}",
            f"Conflicting assertion for {subject}::{predicate}",
        )
        for (subject, predicate), values in grouped_values.items()
    ]

    return build_dimension_evaluation("consistency", checks)
