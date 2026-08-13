#!/usr/bin/env python3
"""Run the import example with filesystem access restricted to this directory."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
sys.path.insert(0, str(REPOSITORY))

from jinest import resolve_file  # noqa: E402


# None would allow every readable path and [] would deny every import. Resolving
# paths and symlinks before this check prevents "../" from escaping HERE.
rendered = resolve_file(
    HERE / "example.yml",
    output_format="yaml",
    import_roots=[HERE],
)
print(rendered, end="")
