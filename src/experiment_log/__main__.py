"""CLI entry point for experiment log module.

Usage:
    uv run python -m src.experiment_log heuristics --classifier fp
    uv run python -m src.experiment_log record-decision --classifier fp ...
    uv run python -m src.experiment_log update-heuristic --classifier fp ...
"""

import sys

from .cli import main

sys.exit(main())
