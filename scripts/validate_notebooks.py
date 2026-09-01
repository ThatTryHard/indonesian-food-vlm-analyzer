#!/usr/bin/env python
"""Verify generated notebooks are clean JSON with syntactically valid code cells."""

from __future__ import annotations

import ast
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def validate_notebook(path: Path) -> tuple[int, int]:
    with path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    if notebook.get("nbformat") != 4:
        raise ValueError(f"{path}: expected notebook format 4")
    code_count = 0
    for cell_index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        code_count += 1
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise ValueError(f"{path}: code cell {cell_index} contains stale execution state")
        source = "".join(cell.get("source", []))
        ast.parse(source, filename=f"{path}:cell-{cell_index}")
    return len(notebook.get("cells", [])), code_count


def main() -> None:
    paths = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))
    if not paths:
        raise FileNotFoundError("No generated notebooks found")
    for path in paths:
        cells, code_cells = validate_notebook(path)
        print(f"{path}: {cells} cells; {code_cells} code cells; clean")


if __name__ == "__main__":
    main()
