#!/usr/bin/env python
"""Render README-facing results from one machine-readable metrics artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.artifacts import render_results_markdown  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output", default=PROJECT_ROOT / "reports" / "RESULTS.md", type=Path)
    args = parser.parse_args()
    with args.metrics.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_results_markdown(metrics)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
