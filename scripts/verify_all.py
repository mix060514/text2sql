#!/usr/bin/env python3
"""Run deterministic boundary and orchestration tests without a model service."""
import sys
from pathlib import Path
import unittest

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project / "src"))

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(str(project / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
