"""Optional live-agent runners.

These are opt-in: ``python -m agent_k`` still runs fully offline by default.
Importing from :mod:`agent_k.agents.ollama` triggers a ``localhost`` HTTP
call only when you actually invoke ``run_scenario`` / ``complete``.
"""

from __future__ import annotations

__all__: list[str] = []
