"""Declarative capability profiles for the agent under evaluation.

An :class:`AgentProfile` describes what the agent-under-test *claims to
support*. Agent K uses profiles as input-only, declarative auto-calibration:
profiles select which evaluation dimensions apply, but never alter any
scoring math or heuristic. This keeps the evaluator deterministic and
auditable while letting reports fairly compare agents with different
capability surfaces.

Design rules:

- Profiles are declared by the caller, not learned by Agent K.
- Profiles are captured in the evaluation report so two runs under
  different profiles are never silently compared as equals.
- A profile can *exclude* a dimension (marking it not-applicable) but can
  never *relax* a dimension's score, cap, severity, or heuristic.
- The default behaviour (``profile=None``) is intentionally the most
  conservative: every dimension runs and uncalibrated confidence is a
  medium-severity finding.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentProfile:
    """Declared capabilities of the agent under evaluation.

    Attributes:
        emits_confidence: When ``False``, the agent is declared to not emit a
            calibrated numeric confidence. The ``confidence_calibration``
            dimension is excluded from the evaluation (not scored, not
            averaged, not flagged). This is the honest outcome for base
            chat LLMs that have no calibration channel; flagging it as a
            finding on every request drowns out real signal.
    """

    emits_confidence: bool = True


#: Profile for uncalibrated base chat LLMs (e.g. local Ollama chat models).
#: These models answer freely but do not produce a calibrated confidence
#: number, so confidence calibration is declared not-applicable rather than
#: failed on every turn.
UNCALIBRATED_CHAT = AgentProfile(emits_confidence=False)


#: Explicit default profile. Use this when you want to be explicit that the
#: agent is expected to emit calibrated confidence (e.g. an internal agent
#: with a structured output schema). Equivalent to passing ``profile=None``.
CALIBRATED_DEFAULT = AgentProfile(emits_confidence=True)
