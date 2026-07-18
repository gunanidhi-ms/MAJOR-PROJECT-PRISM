"""
tests/conftest.py
-----------------
Shared pytest configuration and fixtures for Phase 3 test suite.
"""

import os
import sys
from pathlib import Path

# Make the phase3_reporting root importable regardless of where pytest is run from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set SKIP_LLM globally so no test ever needs a running Ollama instance.
os.environ.setdefault("SKIP_LLM", "true")
os.environ.setdefault("REPORTS_DIR", "reports_test")
