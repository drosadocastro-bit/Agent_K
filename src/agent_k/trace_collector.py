from __future__ import annotations

from agent_k.models import Scenario, Trace, TraceMetadata


class TraceCollector:
    """Builds a trace from deterministic scenario data."""

    def collect(self, scenario: Scenario) -> Trace:
        metadata = TraceMetadata(
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            scenario_description=scenario.description,
            tags=scenario.tags,
        )
        return Trace(prompt=scenario.prompt, output=scenario.output, metadata=metadata)
