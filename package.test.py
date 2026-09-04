#!/usr/bin/env python3
"""Smoke tests for the wheel-only package import surface."""

from __future__ import annotations

import tempfile
from pathlib import Path

from jinest import collections, documents, files, path, runtime, serialization
from jinest.helpers.collections import union
from jinest.helpers.documents import export_yaml, load_yaml
from jinest.helpers.files import file_path
from jinest.helpers.path import path_of
from jinest.helpers.runtime import node
from jinest.helpers.serialization import json_normalize


def main() -> None:
    assert serialization.json_normalize({"x": 1}) == {"x": 1}
    assert json_normalize({"x": 2}) == {"x": 2}
    assert union([1, 2], [2, 3]) == [1, 2, 3]

    resolver = __import__("jinest").Resolver({"branch": {"value": 7}})
    assert path.at(resolver.root.path.branch.value) == 7
    assert path_of(resolver.root.branch).info.segments == ("branch",)
    synthetic = node({"value": 8}, resolver=resolver)
    assert runtime.resolve(synthetic) == {"value": 8}

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        source = folder / "source.yml"
        source.write_text("x: 3\n", encoding="utf-8")
        assert documents.load_yaml(source) == {"x": 3}
        assert load_yaml(source) == {"x": 3}
        target = folder / "out.yml"
        exported = export_yaml({"x": 4}, target)
        assert target.read_text(encoding="utf-8") == exported
        assert files.file_path(target, resolver=resolver) == target.resolve()
        assert file_path(target, resolver=resolver) == target.resolve()

    assert collections.union([1], [1]) == [1]
    print("package imports: OK")


if __name__ == "__main__":
    main()
