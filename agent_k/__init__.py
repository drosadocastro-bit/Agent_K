from __future__ import annotations

from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent
_SRC_PACKAGE = _PACKAGE_ROOT.parent / "src" / "agent_k"

__path__ = [str(_PACKAGE_ROOT), str(_SRC_PACKAGE)]
__version__ = "0.1.0"

from agent_k.api import evaluate
from agent_k.openclaw import OpenClawCapture

__all__ = ["__version__", "OpenClawCapture", "evaluate"]
