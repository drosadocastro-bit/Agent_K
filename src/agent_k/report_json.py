from __future__ import annotations

import json

from agent_k.models import RunReport


def render_json_report(report: RunReport) -> str:
    return json.dumps(report.to_dict(), indent=2)
