"""Ensures the repo root is on sys.path so `backend.*` and `ml.*` imports
work when pytest is invoked from the project root (the normal case)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
