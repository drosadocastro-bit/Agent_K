from __future__ import annotations

from dataclasses import dataclass

from agent_k.models import DimensionEvaluation, DimensionScore, IntegrityFlag, Severity


@dataclass(frozen=True)
class ScoreCheck:
    passed: bool
    success_detail: str
    failure_detail: str


def check(passed: bool, success_detail: str, failure_detail: str) -> ScoreCheck:
    return ScoreCheck(
        passed=passed,
        success_detail=success_detail,
        failure_detail=failure_detail,
    )


def build_flag(flag_type: str, severity: Severity, detail: str) -> IntegrityFlag:
    return IntegrityFlag(type=flag_type, severity=severity, detail=detail)


def build_dimension_evaluation(
    name: str,
    checks: list[ScoreCheck],
    flags: tuple[IntegrityFlag, ...] = (),
) -> DimensionEvaluation:
    if not checks:
        checks = [check(True, "No checks were configured.", "No checks were configured.")]

    passed = [item.success_detail for item in checks if item.passed]
    failed = [item.failure_detail for item in checks if not item.passed]
    score = round(sum(1 for item in checks if item.passed) / len(checks), 3)

    if failed:
        rationale = "; ".join(failed)
        evidence = tuple(passed + failed)
    else:
        rationale = "All checks passed."
        evidence = tuple(passed)

    return DimensionEvaluation(
        score=DimensionScore(name=name, score=score, rationale=rationale, evidence=evidence),
        flags=flags,
    )
