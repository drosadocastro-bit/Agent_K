from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest
import shutil


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def tmp_path():
    base = ROOT / ".tmp"
    base.mkdir(exist_ok=True)
    path = base / f"tmp-{uuid4().hex}"
    path.mkdir()

    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
