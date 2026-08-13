#!/usr/bin/env python3
"""Run the Python-only Jinest options used by this example."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
sys.path.insert(0, str(REPOSITORY))

from jinest import JinestWarningError, Resolver  # noqa: E402


def surround(value: object, left: str, right: str) -> str:
    return f"{left}{value}{right}"


data = yaml.safe_load((HERE / "example.yml").read_text(encoding="utf-8"))
resolver = Resolver(
    data,
    in_place=True,
    strict=False,
    globals={"double": lambda value: value * 2},
    filters={"surround": surround},
    source_path=HERE / "example.yml",
    debug=True,
)

# A valid field can be read without eagerly resolving the entire document.
assert resolver.root.doubled == 10

result = resolver.resolve()
assert result is data  # in_place=True preserves the original mapping object.

# Messages stay available as structured values even after stderr emission.
assert {message.level for message in resolver.messages} == {"warning", "hint"}
assert all(
    message.msg and message.path and message.file
    for message in resolver.messages
)

# emit_messages=False keeps stderr quiet without discarding diagnostics.
# Warnings can independently make resolution fail; hints alone never do.
try:
    Resolver(
        {"choice": "literal", "choice$": "missing"},
        emit_messages=False,
        treat_warnings_as_errors=True,
    ).resolve()
except JinestWarningError:
    pass
else:
    raise AssertionError("treat_warnings_as_errors did not reject a warning")

print(yaml.safe_dump(result, allow_unicode=True, sort_keys=False), end="")
