#!/usr/bin/env python3
"""Run the Python-only Jinest options used by this example."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
sys.path.insert(0, str(REPOSITORY))

from jinest import JinestWarningError, Resolver, helpers  # noqa: E402


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

# Python API: synthetic lazy bindings, explicit globals and targeted resolve.
resolver.update_globals({"factor": 10})
synthetic = resolver.node({"value$": "input * factor"}, vars={"input": 5})
assert synthetic.value == 50
assert resolver.resolve(resolver.root.path.value) == 5
assert resolver.eval("factor") == 10
assert resolver.root.factory.fn()("web").kind == "web"

# The helper namespaces are Python-only facades over the same runtime.
assert helpers.path.at(resolver.root.path.value) == 5  # owner inferred from PathRef
assert helpers.runtime.render("{{ factor }}", resolver=resolver) == "10"
assert helpers.runtime.eval("value", context=resolver.root.path) == 5
assert resolver.resolve(helpers.runtime.literal({"x$": "literal"}), vars={}) == {"x$": "literal"}
assert helpers.serialization.from_json('{"x": 1}') == {"x": 1}
assert helpers.collections.combine({"x": {"a": 1}}, {"x": {"b": 2}}, recursive=True) == {"x": {"a": 1, "b": 2}}
assert helpers.documents.load_yaml(HERE / "example.yml")["value"] == 5
assert helpers.files.read_lines(HERE / "example.yml", resolver=resolver)
plugin_defaults = helpers.documents.import_tree({"port": 8080}, resolver=resolver, source="plugin://example/defaults")
assert plugin_defaults.file == "plugin://example/defaults"
with tempfile.TemporaryDirectory() as temporary:
    destination = Path(temporary) / "synthetic.yaml"
    assert "value: 50" in helpers.documents.export_yaml(synthetic, destination)
    assert destination.exists()

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
