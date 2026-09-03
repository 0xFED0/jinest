#!/usr/bin/env python3
"""Run the Jinja stdlib example and a restricted-namespace capability check."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from jinest import Resolver  # noqa: E402

data = yaml.safe_load((HERE / "example.yml").read_text(encoding="utf-8"))
result = Resolver(data, source_path=HERE / "example.yml", emit_messages=False).resolve()

# Namespace configuration is capability reduction for Jinest-added helpers.
restricted = Resolver(
    {"read_text": "ordinary field", "seen$": "read_text"},
    stdlib_exclude={"files", "documents"},
    emit_messages=False,
)
assert restricted.resolve() == {"read_text": "ordinary field", "seen": "ordinary field"}
assert "files" not in restricted.stdlib.enabled
assert "documents" not in restricted.stdlib.enabled

print(yaml.safe_dump(result, allow_unicode=True, sort_keys=False), end="")
