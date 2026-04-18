from __future__ import annotations

import json

from agent_k.openclaw_models import OpenClawEvaluation



def render_openclaw_json(evaluation: OpenClawEvaluation) -> str:
    return json.dumps(evaluation.to_dict(), indent=2)
