#!/usr/bin/env python3
"""Regression tests for Jinest 0.8.1.

Run:
    python jinest.test.py

By default the test loader imports ``jinest.py`` from the same directory.
Set JINEST_MODULE=/path/to/jinest.py to test another copy.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def _load_jinest() -> Any:
    candidates: list[Path] = []
    if os.environ.get("JINEST_MODULE"):
        candidates.append(Path(os.environ["JINEST_MODULE"]).expanduser())
    candidates.extend(
        [
            HERE / "jinest.py",
        ]
    )

    module_path = next((p.resolve() for p in candidates if p.is_file()), None)
    if module_path is None:
        searched = "\n  ".join(str(p) for p in candidates)
        raise RuntimeError(f"Could not find Jinest module. Searched:\n  {searched}")

    module_name = "jinest_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


jinest = _load_jinest()

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


class JinestCoreTests(unittest.TestCase):
    def test_version(self) -> None:
        self.assertEqual(jinest.__version__, "0.8.1")

    def test_scalar_roots_and_extended_scalars(self) -> None:
        values = [None, True, 42, 3.5, "text", b"\x00A\xff", date(2026, 8, 2)]
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(jinest.resolve(value), value)

        for source in ["null", "true", "42", "3.5", '"text"']:
            with self.subTest(source=source):
                self.assertEqual(
                    json.loads(jinest.resolve_text(source, format="json")),
                    json.loads(source),
                )

        self.assertIsNone(
            json.loads(
                jinest.resolve_text("", format="yaml", output_format="json")
            )
        )

    def test_unsupported_scalars_follow_strict_mode(self) -> None:
        with self.assertRaisesRegex(jinest.JinestError, "Unsupported scalar value"):
            jinest.resolve(object())
        self.assertIsNone(jinest.resolve(object(), strict=False))

    def test_text_and_native_fields(self) -> None:
        result = jinest.resolve(
            {
                "example": {
                    "x": 2,
                    "message@": "x={{ x }}, doubled={{ x * 2 }}",
                    "sum$": "x + 3",
                    "object$": "{'value': x, 'enabled': true}",
                }
            }
        )
        self.assertEqual(result["example"]["message"], "x=2, doubled=4")
        self.assertEqual(result["example"]["sum"], 5)
        self.assertEqual(result["example"]["object"], {"value": 2, "enabled": True})

    def test_hidden_fields_are_template_private_and_take_priority(self) -> None:
        resolver = jinest.Resolver(
            {
                "price": 100,
                ".tax$": "price * 0.2",
                "tax": "public tax value",
                "total$": "price + tax",
                ".private_label@": "private {{ price }}",
                "nested": {
                    ".label@": "private {{ name }}",
                    "name": "Jinest",
                    "label": "public label",
                    "message@": "{{ label }}",
                },
            }
        )
        self.assertEqual(resolver.root.tax, 20.0)
        self.assertEqual(
            resolver.resolve(),
            {
                "price": 100,
                "tax": "public tax value",
                "total": 120.0,
                "nested": {
                    "name": "Jinest",
                    "label": "public label",
                    "message": "private Jinest",
                },
            },
        )

        self.assertEqual(
            jinest.resolve(
                {
                    "base": {".port": 8000},
                    "service": {
                        "<<$": "root.base",
                        "port": 9000,
                        "effective$": "port * 2",
                    },
                    ".answer^": "% return 41 + 1",
                    "next_answer$": "answer + 1",
                }
            ),
            {
                "base": {},
                "service": {"port": 9000, "effective": 16000},
                "next_answer": 43,
            },
        )

    def test_local_priority_concrete_then_native_then_text(self) -> None:
        result = jinest.resolve(
            {
                "native_wins": {
                    "value@": "{{ missing.deep.value }}",
                    "value$": "40 + 2",
                },
                "concrete_wins": {
                    "value": "literal",
                    "value$": "missing.deep.value",
                    "value@": "{{ missing.deep.value }}",
                },
            }
        )
        self.assertEqual(result["native_wins"], {"value": 42})
        self.assertEqual(result["concrete_wins"], {"value": "literal"})

    def test_unrelated_field_remains_lazy(self) -> None:
        resolver = jinest.Resolver(
            {
                "ok$": "21 * 2",
                "bad@": "{{ missing.deep.value }}",
            }
        )
        self.assertEqual(resolver.root.ok, 42)
        with self.assertRaises(jinest.JinestTemplateError):
            resolver.resolve()

    def test_native_expression_rejects_template_delimiters(self) -> None:
        with self.assertRaisesRegex(
            jinest.JinestTemplateError,
            "must not use Jinja template delimiters",
        ):
            jinest.resolve({"bad$": "{{ 1 + 1 }}"})

    def test_non_strict_undefined_is_none_for_native_and_script(self) -> None:
        result = jinest.resolve(
            {
                "native$": "missing.deep.value",
                "arithmetic$": "missing + 1",
                "script^": "% return missing.deep.value\n",
                "text@": "{{ missing.deep.value }}",
            },
            strict=False,
        )
        self.assertEqual(
            result,
            {
                "native": None,
                "arithmetic": None,
                "script": None,
                "text": "",
            },
        )

    def test_field_cycle_resolves_to_none(self) -> None:
        self.assertEqual(
            jinest.resolve({"cycle": {"a$": "b", "b$": "a"}}),
            {"cycle": {"a": None, "b": None}},
        )

    def test_custom_globals_and_filters(self) -> None:
        result = jinest.resolve(
            {
                "native$": "double(5)",
                "text@": "{{ 4 | triple }}",
            },
            globals={"double": lambda x: x * 2},
            filters={"triple": lambda x: x * 3},
        )
        self.assertEqual(result, {"native": 10, "text": "12"})

    def test_in_place_mapping(self) -> None:
        data = {"x": 2, "y$": "x + 1"}
        result = jinest.resolve(data, in_place=True)
        self.assertIs(result, data)
        self.assertEqual(data, {"x": 2, "y": 3})


class JinestLayerTests(unittest.TestCase):
    def test_layer_order_defaults_local_overrides(self) -> None:
        data = {
            "defaults1": {"rank": "d1", "d1": True},
            "defaults2": {"rank": "d2", "d2": True},
            "overrides1": {"rank": "o1", "o1": True},
            "overrides2": {"rank": "o2", "o2": True},
            "example": {
                # Deliberately scrambled source order. Numeric order controls
                # each family independently.
                "<<2!$": "root.overrides2",
                "<<2$": "root.defaults2",
                "rank": "local",
                "<<1!$": "root.overrides1",
                "<<1$": "root.defaults1",
            },
        }
        result = jinest.resolve(data)["example"]
        self.assertEqual(result["rank"], "o2")
        self.assertTrue(result["d1"])
        self.assertTrue(result["d2"])
        self.assertTrue(result["o1"])
        self.assertTrue(result["o2"])

    def test_default_layers_do_not_override_local(self) -> None:
        result = jinest.resolve(
            {
                "base": {"x": 1, "y": 2},
                "target": {"<<$": "root.base", "x": 10},
            }
        )
        self.assertEqual(result["target"], {"x": 10, "y": 2})

    def test_override_layers_override_local(self) -> None:
        result = jinest.resolve(
            {
                "override": {"x": 99, "y": 2},
                "target": {"x": 10, "<<!$": "root.override"},
            }
        )
        self.assertEqual(result["target"], {"x": 99, "y": 2})

    def test_transitive_lazy_layers(self) -> None:
        result = jinest.resolve(
            {
                "A": {"x": 1},
                "B": {"<<$": "root.A"},
                "C": {"<<$": "root.B"},
            }
        )
        self.assertEqual(result["B"], {"x": 1})
        self.assertEqual(result["C"], {"x": 1})

    def test_merge_requires_mapping(self) -> None:
        with self.assertRaises(jinest.JinestMergeError):
            jinest.resolve({"target": {"<<$": "123"}})

    def test_recursive_merge_layer_behaves_as_empty(self) -> None:
        result = jinest.resolve({"A": {"<<$": "root.A", "x": 1}})
        self.assertEqual(result, {"A": {"x": 1}})


class JinestPrototypeTests(unittest.TestCase):
    def _prototype_data(self) -> dict[str, Any]:
        return {
            "class": {
                "parent_var": 1,
                "constant": 10,
                "prototype": {
                    "var": 0,
                    "A$": "_.parent_var",
                    "B$": "var",
                    "C$": "root.class.constant",
                    "where@": "{{ path }}",
                },
            },
            "inherited": {
                "parent_var": 2,
                "instance": {"<<$": "global_root.class.prototype", "var": 1},
                "other$": "global_root.class.prototype",
            },
        }

    def test_prototype_rebinds_relative_context(self) -> None:
        result = jinest.resolve(self._prototype_data())
        self.assertEqual(
            result["inherited"]["instance"],
            {
                "var": 1,
                "A": 2,
                "B": 1,
                "C": 10,
                "where": "global_root.inherited.instance",
            },
        )
        self.assertEqual(
            result["inherited"]["other"],
            {
                "var": 0,
                "A": 2,
                "B": 0,
                "C": 10,
                "where": "global_root.inherited.other",
            },
        )

    def test_source_cache_does_not_leak_into_bound_instance(self) -> None:
        resolver = jinest.Resolver(self._prototype_data())
        # Resolve in the source location first.
        self.assertEqual(resolver.root["class"].prototype.A, 1)
        self.assertEqual(resolver.root["class"].prototype.where, "global_root.class.prototype")
        # A destination-bound copy must still use its own context/cache.
        self.assertEqual(resolver.global_root.inherited.instance.A, 2)
        self.assertEqual(
            resolver.global_root.inherited.instance.where,
            "global_root.inherited.instance",
        )

    def test_same_prototype_has_independent_destination_paths(self) -> None:
        resolver = jinest.Resolver(
            {
                "prototype": {"where$": "path"},
                "left$": "root.prototype",
                "right$": "root.prototype",
            }
        )
        self.assertEqual(str(resolver.root.left.where), "global_root.left")
        self.assertEqual(str(resolver.root.right.where), "global_root.right")


class JinestArrayTests(unittest.TestCase):
    def test_native_and_text_array_items(self) -> None:
        result = jinest.resolve(
            {
                "var1": 7,
                "var2": 9,
                "native_array$": [
                    "var1",
                    "root.var2",
                    "1",
                    "true",
                    5,
                    None,
                    "path",
                ],
                "text_array@": [
                    "{{ var1 }}",
                    "v={{ root.var2 }}",
                    1,
                    True,
                    "{{ path }}",
                ],
            }
        )
        self.assertEqual(
            result["native_array"],
            [7, 9, 1, True, 5, None, "global_root.native_array[6]"],
        )
        self.assertEqual(
            result["text_array"],
            ["7", "v=9", 1, True, "global_root.text_array[4]"],
        )

    def test_array_is_lazy_per_item(self) -> None:
        resolver = jinest.Resolver(
            {
                "items$": [
                    "40 + 2",
                    "missing.deep.value",
                ]
            }
        )
        self.assertEqual(resolver.root.items[0], 42)
        with self.assertRaises(jinest.JinestTemplateError):
            _ = resolver.root.items[1]

    def test_list_returned_by_expression_is_not_reexecuted(self) -> None:
        result = jinest.resolve({"ready_array$": "['var1', 'root.var2']"})
        self.assertEqual(result["ready_array"], ["var1", "root.var2"])

    def test_paths_through_arrays_and_non_identifier_keys(self) -> None:
        result = jinest.resolve(
            {
                "items": [
                    {"obj": {"where$": "path"}},
                    {"obj": {"where@": "{{ path }}"}},
                ],
                "some-key": {"where$": "path"},
            }
        )
        self.assertEqual(result["items"][0]["obj"]["where"], "global_root.items[0].obj")
        self.assertEqual(result["items"][1]["obj"]["where"], "global_root.items[1].obj")
        self.assertEqual(result["some-key"]["where"], "global_root['some-key']")

    def test_sequence_indexing_slicing_and_negative_index(self) -> None:
        resolver = jinest.Resolver({"items$": ["1", "2", "3"]})
        self.assertEqual(resolver.root.items[-1], 3)
        self.assertEqual(resolver.root.items[0:2], [1, 2])
        with self.assertRaises(IndexError):
            _ = resolver.root.items[10]


@unittest.skipUnless(yaml is not None, "PyYAML is required for YAML tests")
class JinestFormatAndImportTests(unittest.TestCase):
    def test_resolve_text_json_and_yaml(self) -> None:
        json_result = json.loads(
            jinest.resolve_text(
                json.dumps({"x": 1, "y$": "x + 1"}),
                format="json",
            )
        )
        self.assertEqual(json_result, {"x": 1, "y": 2})

        yaml_result = yaml.safe_load(
            jinest.resolve_text(
                "x: 1\ny$: 'x + 1'\n",
                format="yaml",
            )
        )
        self.assertEqual(yaml_result, {"x": 1, "y": 2})

    def test_yaml_extended_scalars_normalize_to_json(self) -> None:
        rendered = jinest.resolve_text(
            "date: 2026-08-02\npayload: !!binary AP9B\n",
            format="yaml",
            output_format="json",
        )
        self.assertIn('"date": "2026-08-02"', rendered)
        self.assertIn(
            '"payload": "\\u0000\\u00FF\\u0041"',
            rendered,
        )
        self.assertEqual(
            json.loads(rendered),
            {"date": "2026-08-02", "payload": "\x00ÿA"},
        )

        root_bytes = jinest.resolve_text(
            "!!binary AP9B\n",
            format="yaml",
            output_format="json",
        )
        self.assertEqual(root_bytes, '"\\u0000\\u00FF\\u0041"')

    def test_yaml_json_import_globals_aliases_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "base.yaml").write_text(
                """
constant: 10
prototype:
  local: 3
  absolute$: root.constant
  relative$: _.parent_var
  where@: "{{ path }}"
""".lstrip(),
                encoding="utf-8",
            )
            (folder / "data.json").write_text(
                json.dumps({"value": 21, "double$": "root.value * 2"}),
                encoding="utf-8",
            )
            (folder / "main.yaml").write_text(
                """
parent_var: 5
instance:
  <<$: import_yaml('base.yaml').prototype
alias_value$: import('base.yaml').constant
json_obj$: import_json('data.json')
json_value$: import_json('data.json').double
via_yaml_filter$: '"base.yaml" | import | attr("constant")'
via_json_filter$: '"data.json" | import_json | attr("double")'
""".lstrip(),
                encoding="utf-8",
            )

            result = json.loads(
                jinest.resolve_file(folder / "main.yaml", output_format="json")
            )
            self.assertEqual(result["instance"]["absolute"], 10)
            self.assertEqual(result["instance"]["relative"], 5)
            self.assertEqual(result["instance"]["where"], "global_root.instance")
            self.assertEqual(result["alias_value"], 10)
            self.assertEqual(result["json_obj"]["double"], 42)
            self.assertEqual(result["json_value"], 42)
            self.assertEqual(result["via_yaml_filter"], 10)
            self.assertEqual(result["via_json_filter"], 42)

    def test_imported_root_is_preserved_while_destination_path_rebinds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "library.yaml").write_text(
                """
constant: 17
prototype:
  absolute$: root.constant
  relative$: _.local
  where$: path
""".lstrip(),
                encoding="utf-8",
            )
            (folder / "main.yaml").write_text(
                """
container:
  local: 23
  instance:
    <<$: import('library.yaml').prototype
""".lstrip(),
                encoding="utf-8",
            )
            result = json.loads(
                jinest.resolve_file(folder / "main.yaml", output_format="json")
            )
            self.assertEqual(
                result["container"]["instance"],
                {
                    "absolute": 17,
                    "relative": 23,
                    "where": "global_root.container.instance",
                },
            )

    def test_nested_relative_imports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "templates").mkdir()
            (folder / "shared").mkdir()
            (folder / "shared" / "common.yaml").write_text(
                "value: 12\ndouble$: root.value * 2\n",
                encoding="utf-8",
            )
            (folder / "templates" / "base.yaml").write_text(
                "common$: import('../shared/common.yaml')\n",
                encoding="utf-8",
            )
            (folder / "main.yaml").write_text(
                "base$: import('templates/base.yaml')\n",
                encoding="utf-8",
            )
            result = json.loads(
                jinest.resolve_file(folder / "main.yaml", output_format="json")
            )
            self.assertEqual(result["base"]["common"]["double"], 24)

    def test_import_roots_restrict_imports_and_propagate_to_children(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            project = folder / "project"
            (project / "templates").mkdir(parents=True)
            (project / "shared").mkdir()
            outside = folder / "outside"
            outside.mkdir()
            (project / "shared" / "common.yaml").write_text(
                "value: 12\n",
                encoding="utf-8",
            )
            (project / "templates" / "base.yaml").write_text(
                "common$: import('../shared/common.yaml')\n",
                encoding="utf-8",
            )
            (project / "main.yaml").write_text(
                "base$: import('templates/base.yaml')\n",
                encoding="utf-8",
            )
            (outside / "secret.yaml").write_text("value: secret\n", encoding="utf-8")
            (project / "escape.yaml").write_text(
                "secret$: import('../outside/secret.yaml')\n",
                encoding="utf-8",
            )

            allowed = json.loads(
                jinest.resolve_file(
                    project / "main.yaml",
                    output_format="json",
                    import_roots=[project],
                )
            )
            self.assertEqual(allowed, {"base": {"common": {"value": 12}}})

            with self.assertRaisesRegex(
                jinest.JinestImportError,
                "outside permitted roots",
            ):
                jinest.resolve_file(
                    project / "escape.yaml",
                    output_format="json",
                    import_roots=[project],
                )
            with self.assertRaises(jinest.JinestImportError):
                jinest.resolve_file(
                    project / "main.yaml",
                    output_format="json",
                    import_roots=[],
                )

    def test_import_cycle_resolves_current_path_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "a.yaml").write_text(
                "other$: import('b.yaml')\n",
                encoding="utf-8",
            )
            (folder / "b.yaml").write_text(
                "back$: import('a.yaml')\n",
                encoding="utf-8",
            )
            result = json.loads(
                jinest.resolve_file(folder / "a.yaml", output_format="json")
            )
            self.assertEqual(result, {"other": {"back": None}})

    def test_missing_import_is_wrapped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "main.yaml").write_text(
                "value$: import('missing.yaml')\n",
                encoding="utf-8",
            )
            with self.assertRaises(jinest.JinestImportError):
                jinest.resolve_file(folder / "main.yaml", output_format="json")


class JinestPathAndMetadataTests(unittest.TestCase):
    def test_node_metadata_and_real_key_collisions(self) -> None:
        resolver = jinest.Resolver(
            {
                "obj": {
                    "path": "real-path",
                    "source_path": "real-source-path",
                    "root": "real-root",
                    "file": "real-file",
                    "meta_path@": "{{ context.path }}",
                    "meta_source_path@": "{{ context.source_path }}",
                    "meta_root_path@": "{{ context.root.path }}",
                    "meta_file$": "context.file",
                    "real_path$": 'context["path"]',
                    "real_root$": 'context["root"]',
                    "real_file$": 'context["file"]',
                }
            }
        )
        obj = resolver.root.obj
        self.assertIsInstance(obj.path, jinest.PathRef)
        self.assertEqual(str(obj.path), "global_root.obj")
        self.assertEqual(str(obj.source_path), "root.obj")
        self.assertEqual(str(obj.root.path), "root")
        self.assertIsNone(obj.file)
        self.assertEqual(obj["path"], "real-path")

        result = resolver.resolve()["obj"]
        self.assertEqual(result["meta_path"], "global_root.obj")
        self.assertEqual(result["meta_source_path"], "root.obj")
        self.assertEqual(result["meta_root_path"], "root")
        self.assertIsNone(result["meta_file"])
        self.assertEqual(result["real_path"], "real-path")
        self.assertEqual(result["real_root"], "real-root")
        self.assertEqual(result["real_file"], "real-file")

    def test_pathref_navigation_absolute_at_get_and_relative(self) -> None:
        result = jinest.resolve(
            {
                "target": {"value": 42, "file": "payload"},
                "calc": {
                    "absolute@": "{{ path._.target.value.absolute }}",
                    "value$": "at(path._.target.value)",
                    "safe$": "get(path._.target.missing, 99)",
                    "relative@": (
                        "{{ relative_path(path_of(global_root.target).value, context) }}"
                    ),
                    "relative_value$": (
                        "context[relative_path(path_of(global_root.target).value, context)]"
                    ),
                    "normalized@": (
                        "{{ normalize_path(\"global_root.target['value']\") }}"
                    ),
                    "path_file_value$": "at(path._.target.file)",
                    "path_absolute_key$": 'at(path._.target["file"])',
                },
            }
        )["calc"]
        self.assertEqual(result["absolute"], "global_root.target.value")
        self.assertEqual(result["value"], 42)
        self.assertEqual(result["safe"], 99)
        self.assertEqual(result["relative"], "_.target.value")
        self.assertEqual(result["relative_value"], 42)
        self.assertEqual(result["normalized"], "global_root.target.value")
        self.assertEqual(result["path_file_value"], "payload")
        self.assertEqual(result["path_absolute_key"], "payload")

    def test_node_indexing_reanchors_relative_path(self) -> None:
        resolver = jinest.Resolver(
            {
                "left": {"value": 1},
                "right": {
                    "value": 2,
                    "picked$": (
                        "global_root.left[relative_path(path_of(global_root.right).value, "
                        "global_root.right)]"
                    ),
                },
            }
        )
        self.assertEqual(resolver.root.right.picked, 1)

    def test_path_helpers_reject_cross_root_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "library.yaml").write_text(
                "prototype:\n"
                "  bad$: relative_path(path_of(root.prototype), context)\n",
                encoding="utf-8",
            )
            (folder / "main.yaml").write_text(
                "instance:\n"
                "  <<$: import('library.yaml').prototype\n",
                encoding="utf-8",
            )
            with self.assertRaises(jinest.JinestPathError):
                jinest.resolve_file(folder / "main.yaml", output_format="json")

    def test_imported_origin_root_global_root_and_file_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            library = folder / "library.yaml"
            library.write_text(
                "constant: 17\n"
                "prototype:\n"
                "  local: 3\n"
                "  root_value$: root.constant\n"
                "  origin_value$: origin.local\n"
                "  context_path@: '{{ context.path }}'\n"
                "  origin_path@: '{{ origin.path }}'\n"
                "  source_path@: '{{ source_path_of(context) }}'\n"
                "  global_path@: '{{ global_root.path }}'\n"
                "  root_file$: root.file\n"
                "  origin_file$: source_file(origin)\n"
                "  source_root_path@: '{{ root_of(origin).path }}'\n",
                encoding="utf-8",
            )
            (folder / "main.yaml").write_text(
                "instance:\n"
                "  <<$: import('library.yaml').prototype\n",
                encoding="utf-8",
            )
            result = json.loads(
                jinest.resolve_file(folder / "main.yaml", output_format="json")
            )["instance"]
            self.assertEqual(result["root_value"], 17)
            self.assertEqual(result["origin_value"], 3)
            self.assertEqual(result["context_path"], "global_root.instance")
            self.assertEqual(result["origin_path"], "root.prototype")
            self.assertEqual(result["source_path"], "root.instance")
            self.assertEqual(result["global_path"], "global_root")
            self.assertEqual(result["root_file"], str(library.resolve()))
            self.assertEqual(result["origin_file"], str(library.resolve()))
            self.assertEqual(result["source_root_path"], "root")

    def test_at_can_return_nodes_and_scalar_fields(self) -> None:
        resolver = jinest.Resolver(
            {
                "obj": {"value": 7},
                "node$": "at(path_of(global_root.obj))",
                "scalar$": "at(path_of(global_root.obj).value)",
            }
        )
        self.assertEqual(resolver.root.node.value, 7)
        self.assertEqual(resolver.root.scalar, 7)


class JinestScriptTests(unittest.TestCase):
    def test_multiline_script_returns_native_value(self) -> None:
        result = jinest.resolve(
            {
                "x": 4,
                "value^": (
                    "% set doubled = x * 2\n"
                    "% if doubled > 5\n"
                    "% return {'value': doubled, 'large': true}\n"
                    "% endif\n"
                    "% return {'value': doubled, 'large': false}\n"
                ),
            }
        )
        self.assertEqual(result["value"], {"value": 8, "large": True})

    def test_text_template_percent_lines_remain_literal(self) -> None:
        result = jinest.resolve({"text@": "% this is ordinary text\n{{ 1 + 1 }}"})
        self.assertEqual(result["text"], "% this is ordinary text\n2")

    def test_script_without_return_and_empty_return_are_none(self) -> None:
        result = jinest.resolve(
            {
                "implicit^": "% set x = 1\n",
                "explicit^": "% return\n",
            }
        )
        self.assertEqual(result, {"implicit": None, "explicit": None})

    def test_return_inside_loop_terminates_script(self) -> None:
        result = jinest.resolve(
            {
                "value^": (
                    "% for item in [1, 2, 3]\n"
                    "% if item == 2\n"
                    "% return item * 10\n"
                    "% endif\n"
                    "% endfor\n"
                    "% return 0\n"
                )
            }
        )
        self.assertEqual(result["value"], 20)

    def test_script_priority_over_native_and_text(self) -> None:
        result = jinest.resolve(
            {
                "script_wins": {
                    "value@": "{{ missing.deep.value }}",
                    "value$": "missing.deep.value",
                    "value^": "% return 42\n",
                },
                "concrete_wins": {
                    "value": "literal",
                    "value^": "% return missing.deep.value\n",
                    "value$": "missing.deep.value",
                    "value@": "{{ missing.deep.value }}",
                },
            }
        )
        self.assertEqual(result["script_wins"], {"value": 42})
        self.assertEqual(result["concrete_wins"], {"value": "literal"})

    def test_script_array_is_resolved_per_string_item(self) -> None:
        result = jinest.resolve(
            {
                "x": 5,
                "values^": [
                    "% return x + 1\n",
                    "% set y = x * 2\n% return y\n",
                    3,
                    None,
                ],
            }
        )
        self.assertEqual(result["values"], [6, 10, 3, None])

    def test_script_default_and_override_layers(self) -> None:
        result = jinest.resolve(
            {
                "defaults": {"x": 1, "default": True},
                "overrides": {"x": 3, "override": True},
                "target": {
                    "<<2^": "% return root.defaults\n",
                    "x": 2,
                    "<<1!^": "% return root.overrides\n",
                },
            }
        )["target"]
        self.assertEqual(result["x"], 3)
        self.assertTrue(result["default"])
        self.assertTrue(result["override"])

    def test_script_layer_none_is_empty(self) -> None:
        result = jinest.resolve(
            {
                "enabled": False,
                "target": {
                    "<<^": (
                        "% if root.enabled\n"
                        "% return {'x': 1}\n"
                        "% endif\n"
                        "% return none\n"
                    ),
                    "y": 2,
                },
            }
        )
        self.assertEqual(result["target"], {"y": 2})

    def test_script_merge_requires_mapping(self) -> None:
        with self.assertRaises(jinest.JinestMergeError):
            jinest.resolve({"target": {"<<^": "% return 123\n"}})

    def test_script_has_full_context_metadata(self) -> None:
        result = jinest.resolve(
            {
                "obj": {
                    "value^": (
                        "% return {\n"
                        "  'context': context.path,\n"
                        "  'origin': origin.path,\n"
                        "  'root': root.path,\n"
                        "  'global': global_root.path,\n"
                        "  'path': path\n"
                        "}\n"
                    )
                }
            }
        )["obj"]["value"]
        self.assertEqual(
            result,
            {
                "context": "global_root.obj",
                "origin": "root.obj",
                "root": "root",
                "global": "global_root",
                "path": "global_root.obj",
            },
        )


class JinestPathEdgeTests(unittest.TestCase):
    def test_string_paths_alias_arguments_and_absolute_conversion(self) -> None:
        result = jinest.resolve(
            {
                "top": {"value": 11},
                "nested": {
                    "child": {
                        "at_string$": 'at("_._.top.value")',
                        "normalized@": '{{ normalize_path("_._.top.value") }}',
                        "absolute@": (
                            '{{ absolute_path(normalize_path("_._.top.value"), root=context) }}'
                        ),
                        "relative@": (
                            '{{ relative_path(path_of(global_root.top).value, '
                            'path=context) }}'
                        ),
                        "relative_absolute@": (
                            '{{ relative_path(path_of(global_root.top).value, '
                            'context).absolute }}'
                        ),
                    }
                },
            }
        )["nested"]["child"]
        self.assertEqual(result["at_string"], 11)
        self.assertEqual(result["normalized"], "_._.top.value")
        self.assertEqual(result["absolute"], "global_root.top.value")
        self.assertEqual(result["relative"], "_._.top.value")
        self.assertEqual(result["relative_absolute"], "global_root.top.value")

    def test_absolute_path_reanchors_existing_relative_path(self) -> None:
        result = jinest.resolve(
            {
                "left": {"value": 1},
                "right": {"value": 2},
                "calc": {
                    "path@": (
                        "{{ absolute_path("
                        "relative_path(path_of(global_root.left).value, "
                        "global_root.left), anchor=global_root.right) }}"
                    ),
                    "value$": (
                        "at(relative_path(path_of(global_root.left).value, "
                        "global_root.left), anchor=global_root.right)"
                    ),
                },
            }
        )["calc"]
        self.assertEqual(result["path"], "global_root.right.value")
        self.assertEqual(result["value"], 2)

    def test_sequence_nodes_expose_metadata(self) -> None:
        resolver = jinest.Resolver({"items": [{"x": 1}]})
        items = resolver.root.items
        self.assertEqual(str(items.path), "global_root.items")
        self.assertEqual(str(items.source_path), "root.items")
        self.assertIsNone(items.file)
        self.assertEqual(str(items.root.path), "root")

    def test_pathref_absolute_name_is_addressable_with_brackets(self) -> None:
        result = jinest.resolve(
            {
                "absolute": 19,
                "value$": 'at(path["absolute"])',
            }
        )
        self.assertEqual(result["value"], 19)


class JinestScriptEdgeTests(unittest.TestCase):
    def test_mixed_native_and_script_layer_order(self) -> None:
        result = jinest.resolve(
            {
                "d1": {"rank": "d1"},
                "d2": {"rank": "d2"},
                "o1": {"rank": "o1"},
                "o2": {"rank": "o2"},
                "target": {
                    "<<2$": "root.d2",
                    "<<1^": "% return root.d1\n",
                    "rank": "local",
                    "<<2!$": "root.o2",
                    "<<1!^": "% return root.o1\n",
                },
            }
        )["target"]
        self.assertEqual(result["rank"], "o2")

    def test_script_returned_prototype_rebinds_destination_context(self) -> None:
        result = jinest.resolve(
            {
                "class": {
                    "parent_value": 0,
                    "prototype": {
                        "value": 1,
                        "parent$": "_.parent_value",
                        "where@": "{{ path }}",
                    },
                },
                "container": {
                    "parent_value": 7,
                    "instance^": "% return root.class.prototype\n",
                },
            }
        )["container"]["instance"]
        self.assertEqual(
            result,
            {
                "value": 1,
                "parent": 7,
                "where": "global_root.container.instance",
            },
        )

    def test_script_field_cycle_resolves_to_none(self) -> None:
        result = jinest.resolve(
            {
                "cycle": {
                    "a^": "% return b\n",
                    "b^": "% return a\n",
                }
            }
        )
        self.assertEqual(result["cycle"], {"a": None, "b": None})

    def test_recursive_script_merge_is_empty(self) -> None:
        result = jinest.resolve(
            {
                "target": {
                    "<<^": "% return root.target\n",
                    "x": 1,
                }
            }
        )
        self.assertEqual(result["target"], {"x": 1})

    def test_imported_script_keeps_source_root_and_global_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "library.yaml").write_text(
                "constant: 5\n"
                "prototype:\n"
                "  info^: |\n"
                "    % return {\n"
                "      'constant': root.constant,\n"
                "      'path': path,\n"
                "      'origin': origin.path,\n"
                "      'global': global_root.path\n"
                "    }\n",
                encoding="utf-8",
            )
            (folder / "main.yaml").write_text(
                "instance:\n"
                "  <<$: import('library.yaml').prototype\n",
                encoding="utf-8",
            )
            result = json.loads(
                jinest.resolve_file(folder / "main.yaml", output_format="json")
            )["instance"]["info"]
            self.assertEqual(
                result,
                {
                    "constant": 5,
                    "path": "global_root.instance",
                    "origin": "root.prototype",
                    "global": "global_root",
                },
            )

    def test_script_syntax_errors_are_wrapped(self) -> None:
        with self.assertRaises(jinest.JinestTemplateError):
            jinest.resolve({"bad^": "% if true\n% return 1\n"})

    @unittest.skipIf(yaml is None, "PyYAML is not installed")
    def test_yaml_script_and_script_merge_syntax(self) -> None:
        rendered = jinest.resolve_text(
            "base:\n"
            "  x: 1\n"
            "target:\n"
            "  <<^: |\n"
            "    % return root.base\n"
            "  y^: |\n"
            "    % set value = x + 1\n"
            "    % return value\n",
            format="yaml",
            output_format="json",
        )
        self.assertEqual(json.loads(rendered)["target"], {"x": 1, "y": 2})


if __name__ == "__main__":
    unittest.main(verbosity=2)
