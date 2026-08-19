#!/usr/bin/env python3
"""Regression tests for Jinest.

Run:
    python jinest.test.py

By default the test loader imports ``jinest.py`` from the same directory.
Set JINEST_MODULE=/path/to/jinest.py to test another copy.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import date, time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def _find_jinest_module() -> Path:
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
    return module_path


def _load_jinest(module_path: Path) -> Any:

    module_name = "jinest_under_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


JINEST_MODULE_PATH = _find_jinest_module()
jinest = _load_jinest(JINEST_MODULE_PATH)

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


class JinestTestSuiteContractTests(unittest.TestCase):
    def test_suite_does_not_access_private_jinest_members(self) -> None:
        """Keep the regression suite independent of Jinest implementation details."""
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        private_accesses = sorted(
            (node.lineno, node.attr)
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "jinest"
            and node.attr.startswith("_")
            and node.attr != "__version__"
        )
        self.assertEqual(private_accesses, [])


class JinestCoreTests(unittest.TestCase):
    def test_version(self) -> None:
        self.assertEqual(jinest.__version__, "0.17.0")

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
        class Uncopyable:
            def __deepcopy__(self, memo: dict[int, Any]) -> Any:
                raise RuntimeError("cannot copy")

        for value in (object(), Uncopyable()):
            with self.subTest(value=value, strict=True):
                with self.assertRaisesRegex(
                    jinest.JinestError,
                    "Unsupported scalar value",
                ):
                    jinest.resolve(value)
            with self.subTest(value=value, strict=False):
                self.assertIsNone(jinest.resolve(value, strict=False))

    def test_generated_mapping_invalid_keys_never_degrade_in_non_strict_mode(self) -> None:
        class InvalidKey:
            pass

        invalid_key = InvalidKey()
        with self.assertRaisesRegex(jinest.JinestError, "Unsupported mapping key"):
            jinest.resolve(
                {"value$": "make()"},
                globals={"make": lambda: {invalid_key: 1}},
                strict=False,
                emit_messages=False,
            )
        with self.assertRaisesRegex(jinest.JinestError, "Unsupported mapping key"):
            jinest.resolve({invalid_key: 1}, strict=False, emit_messages=False)

    def test_evaluator_body_types_are_strict(self) -> None:
        self.assertEqual(
            jinest.resolve(
                {
                    "integer$": 42,
                    "boolean$": True,
                    "float^": 3.5,
                    "false^": False,
                    "wrapped": {"<$": 7},
                },
                emit_messages=False,
            ),
            {
                "integer": 42,
                "boolean": True,
                "float": 3.5,
                "false": False,
                "wrapped": 7,
            },
        )

        invalid = [
            {"value$": {"nested": 1}},
            {"value^": {"nested": 1}},
            {"value@": {"nested": 1}},
            {"value@": 42},
            {"value@": True},
            {"values$": [{"nested": 1}]},
            {"values^": [None]},
            {"values@": [42]},
        ]
        for source in invalid:
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                    jinest.JinestTemplateError,
                    "requires a string body",
                ):
                    jinest.resolve(source, emit_messages=False)

        for declaration in (
            {"bad()$": {}},
            {"bad()^": []},
            {"bad()@": 1},
        ):
            with self.subTest(declaration=declaration):
                with self.assertRaisesRegex(
                    jinest.JinestError,
                    "requires a string body",
                ):
                    jinest.Resolver(declaration, emit_messages=False)

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

    def test_field_key_context_variables(self) -> None:
        result = jinest.resolve(
            {
                "native_context$": "{'keyname': keyname, 'effective_key': effective_key, 'keymode': keymode, 'keypath': keypath}",
                "text_context@": "{{ keyname }}|{{ effective_key }}|{{ keymode }}|{{ keypath }}",
                "script_context^": "\n".join(
                    [
                        "% return {",
                        '  "keyname": keyname,',
                        '  "effective_key": effective_key,',
                        '  "keymode": keymode,',
                        '  "keypath": keypath,',
                        "}",
                    ]
                ),
                "array_context": [
                    "=${'keyname': keyname, 'effective_key': effective_key, 'keymode': keymode, 'keypath': keypath}"
                ],
                ".hidden_context$": "{'keyname': keyname, 'effective_key': effective_key, 'keymode': keymode, 'keypath': keypath}",
                "hidden_report$": "hidden_context",
            }
        )
        self.assertEqual(
            result,
            {
                "native_context": {
                    "keyname": "native_context",
                    "effective_key": "native_context$",
                    "keymode": "$",
                    "keypath": "global_root.native_context",
                },
                "text_context": "text_context|text_context@|@|global_root.text_context",
                "script_context": {
                    "keyname": "script_context",
                    "effective_key": "script_context^",
                    "keymode": "^",
                    "keypath": "global_root.script_context",
                },
                "array_context": [
                    {
                        "keyname": "array_context",
                        "effective_key": "array_context",
                        "keymode": "$",
                        "keypath": "global_root.array_context[0].array_context",
                    }
                ],
                "hidden_report": {
                    "keyname": "hidden_context",
                    "effective_key": ".hidden_context$",
                    "keymode": "$",
                    "keypath": "global_root.hidden_context",
                },
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

        # A malformed self declaration inside a suppressed mode alternative
        # must not be parsed during diagnostic collection.
        self.assertEqual(
            jinest.resolve(
                {
                    "value": "literal",
                    "value$": {"<(x x)=": {}},
                },
                emit_messages=False,
            ),
            {"value": "literal"},
        )

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


    def test_in_place_resolve_rebuilds_state_after_root_replacement(self) -> None:
        """A reused mutable root must not retain prior syntax or view caches."""
        data = {"value$": "1"}
        resolver = jinest.Resolver(data, in_place=True, emit_messages=False)

        self.assertEqual(resolver.resolve(), {"value": 1})
        data.clear()
        data["value$"] = "2"

        self.assertEqual(resolver.resolve(), {"value": 2})

    def test_in_place_materialized_raw_keys_are_atomic_and_idempotent(self) -> None:
        data = {
            "literal$`": "not an expression",
            "f()=`": "literal function-looking key",
            "matrix[i=items]=`": "literal compose-looking key",
        }
        resolver = jinest.Resolver(data, in_place=True, emit_messages=False)
        expected = {
            "literal$": "not an expression",
            "f()=": "literal function-looking key",
            "matrix[i=items]=": "literal compose-looking key",
        }

        first = resolver.resolve()
        self.assertIs(first, data)
        self.assertEqual(first, expected)
        self.assertEqual(list(resolver.root), list(expected))
        self.assertIs(resolver.resolve(), data)
        self.assertEqual(data, expected)

    def test_in_place_custom_mapping_needs_no_deepcopy_or_equality(self) -> None:
        class HostileMapping(dict[str, Any]):
            def __deepcopy__(self, memo: dict[int, Any]) -> Any:
                raise RuntimeError("deepcopy must not be called")

            def __eq__(self, other: object) -> bool:
                return True

        data = HostileMapping({"value$": "1"})
        resolver = jinest.Resolver(data, in_place=True, emit_messages=False)

        self.assertIs(resolver.resolve(), data)
        self.assertEqual(dict(data), {"value": 1})
        data.clear()
        data["value$"] = "2"

        self.assertIs(resolver.resolve(), data)
        self.assertEqual(dict(data), {"value": 2})

    def test_failed_in_place_resolution_does_not_mutate_input(self) -> None:
        data = {"kept": 1, "invalid$": "("}
        resolver = jinest.Resolver(data, in_place=True, emit_messages=False)

        with self.assertRaises(jinest.JinestTemplateError):
            resolver.resolve()

        self.assertEqual(data, {"kept": 1, "invalid$": "("})

    def test_in_place_resolve_starts_a_fresh_import_and_diagnostic_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            imported = folder / "value.json"
            imported.write_text('{"value": 1}', encoding="utf-8")
            data = {
                "imported$": "import_json('value.json')",
                "name": 1,
                "name$": "2",
            }
            resolver = jinest.Resolver(
                data,
                base_dir=folder,
                in_place=True,
                emit_messages=False,
            )

            self.assertEqual(
                resolver.resolve(),
                {"imported": {"value": 1}, "name": 1},
            )
            self.assertEqual(
                [message.level for message in resolver.messages],
                ["warning"],
            )

            imported.write_text('{"value": 2}', encoding="utf-8")
            data.clear()
            data["imported$"] = "import_json('value.json')"

            self.assertEqual(resolver.resolve(), {"imported": {"value": 2}})
            self.assertEqual(resolver.messages, [])

    def test_failed_field_layer_and_dynamic_key_accesses_retry_cleanly(self) -> None:
        """Each public lazy cache removes its temporary entry after an error."""
        field_calls = 0

        def flaky_field() -> int:
            nonlocal field_calls
            field_calls += 1
            if field_calls == 1:
                raise RuntimeError("transient field failure")
            return 42

        field = jinest.Resolver(
            {"value$": "flaky_field()"},
            globals={"flaky_field": flaky_field},
            emit_messages=False,
        )
        with self.assertRaises(jinest.JinestTemplateError):
            _ = field.root.value
        self.assertEqual(field.root.value, 42)
        self.assertEqual(field_calls, 2)

        layer_calls = 0

        def flaky_layer() -> bool:
            nonlocal layer_calls
            layer_calls += 1
            if layer_calls == 1:
                raise RuntimeError("transient layer failure")
            return True

        layer = jinest.Resolver(
            {
                "base": {"value": 43},
                "target": {"<<$": "flaky_layer() and root.base"},
            },
            globals={"flaky_layer": flaky_layer},
            emit_messages=False,
        )
        with self.assertRaises(jinest.JinestTemplateError):
            _ = layer.root.target["value"]
        self.assertEqual(layer.root.target["value"], 43)
        self.assertEqual(layer_calls, 2)

        key_calls = 0

        def flaky_key() -> str:
            nonlocal key_calls
            key_calls += 1
            if key_calls == 1:
                raise RuntimeError("transient key failure")
            return "value"

        dynamic = jinest.Resolver(
            {"=$flaky_key()": 44},
            globals={"flaky_key": flaky_key},
            emit_messages=False,
        )
        with self.assertRaises(jinest.JinestTemplateError):
            list(dynamic.root)
        self.assertEqual(dynamic.resolve(), {"value": 44})
        self.assertEqual(key_calls, 2)

class JinestFunctionTests(unittest.TestCase):
    def test_function_modes_namespace_and_structured_returns(self) -> None:
        result = jinest.resolve(
            {
                "square(x)$": "x * x",
                "quote(value, mark='\"')@": "{{ mark }}{{ value }}{{ mark }}",
                "clamp(value, minimum, maximum)^": (
                    "% if value < minimum\n"
                    "%   return minimum\n"
                    "% endif\n"
                    "% if value > maximum\n"
                    "%   return maximum\n"
                    "% endif\n"
                    "% return value\n"
                ),
                "make_result(value)^": (
                    '% return {"value": value, "type": "generated"}\n'
                ),
                "strings": {
                    "upper(value)$": "value | upper",
                    "quote(value)@": "<{{ value }}>",
                },
                "native_result$": "square(5)",
                "text_result@": "{{ quote(square(5)) }}",
                "script_result$": "clamp(15, 0, 10)",
                "struct_result$": 'make_result("hello")',
                "namespace_result@": "{{ strings.quote(strings.upper('hello')) }}",
                "ordinary(value)": "kept as an ordinary field",
            }
        )
        self.assertEqual(
            result,
            {
                "strings": {},
                "native_result": 25,
                "text_result": '"25"',
                "script_result": 10,
                "struct_result": {"value": "hello", "type": "generated"},
                "namespace_result": "<HELLO>",
                "ordinary(value)": "kept as an ordinary field",
            },
        )

    def test_function_arguments_defaults_and_call_site_context(self) -> None:
        result = jinest.resolve(
            {
                "value": 100,
                "defaults": {"factor": 10},
                "identity(value)$": "value",
                "scale(value, factor=context.defaults.factor)$": "value * factor",
                "helpers": {
                    "prefix": "helper-",
                    "format(value)@": "{{ context.prefix }}{{ value }}",
                },
                "identity_result$": "global_root.identity(5)",
                "default_result$": "global_root.scale(5)",
                "named_result$": "global_root.scale(5, factor=3)",
                "output": {
                    "prefix": "output-",
                    "call_site@": "{{ global_root.helpers.format('name') }}",
                },
            }
        )
        self.assertEqual(result["identity_result"], 5)
        self.assertEqual(result["default_result"], 50)
        self.assertEqual(result["named_result"], 15)
        self.assertEqual(result["output"]["call_site"], "output-name")

    def test_structural_functions_rebind_body_at_call_site(self) -> None:
        result = jinest.resolve(
            {
                "declared": "declaration-value",
                "make(x, y=global_root.default, k='generated', v=x * y)=": {
                    "x$": "x",
                    "nested": {
                        "product$": "x * y",
                        "context_path$": "context.path",
                    },
                    "label@": "{{ context.prefix }}{{ x }}",
                    "where@": "{{ path }}",
                    "origin_value$": "origin.declared",
                    "root_value$": "root.declared",
                    "=$k": "=$v",
                },
                "pair(a, b=global_root.default)=": ["=$a", "=@{{ b }}"],
                "default": 10,
                "left": {"prefix": "left-", "result": "=$global_root.make(2)"},
                "right": {"prefix": "right-", "result": "=$global_root.make(3, y=4, k='answer')"},
                "array": "=$pair(7)",
            },
            emit_messages=False,
        )
        self.assertEqual(
            result,
            {
                "declared": "declaration-value",
                "default": 10,
                "left": {
                    "prefix": "left-",
                    "result": {
                        "x": 2,
                        "nested": {
                            "product": 20,
                            "context_path": "global_root.left.result.nested",
                        },
                        "label": "left-2",
                        "where": "global_root.left.result",
                        "origin_value": "declaration-value",
                        "root_value": "declaration-value",
                        "generated": 20,
                    },
                },
                "right": {
                    "prefix": "right-",
                    "result": {
                        "x": 3,
                        "nested": {
                            "product": 12,
                            "context_path": "global_root.right.result.nested",
                        },
                        "label": "right-3",
                        "where": "global_root.right.result",
                        "origin_value": "declaration-value",
                        "root_value": "declaration-value",
                        "answer": 12,
                    },
                },
                "array": [7, "10"],
            },
        )

    def test_structural_function_result_rebinds_each_attachment(self) -> None:
        result = jinest.resolve(
            {
                "make(value)=": {
                    "value$": "value",
                    "path$": "path",
                    "where$": "path",
                    "nested": {"path$": "path"},
                },
                "result^": "\n".join(
                    [
                        "% set x = global_root.make(10)",
                        # Warm the temporary call-site node before attaching it.
                        "% set warmed = x.value",
                        "% set warmed_nested = x.nested.path",
                        "% set tmp = x.where",
                        '% return {"x": x, "y": x, "tmp": tmp}',
                    ]
                ),
            },
            emit_messages=False,
        )
        self.assertEqual(
            result,
            {
                "result": {
                    "x": {
                        "value": 10,
                        "path": "global_root.result.x",
                        "where": "global_root.result.x",
                        "nested": {"path": "global_root.result.x.nested"},
                    },
                    "y": {
                        "value": 10,
                        "path": "global_root.result.y",
                        "where": "global_root.result.y",
                        "nested": {"path": "global_root.result.y.nested"},
                    },
                    "tmp": "global_root.result",
                }
            },
        )

    def test_structural_function_layers_in_body_and_nested_node(self) -> None:
        result = jinest.resolve(
            {
                ".root_native0": {"native0": True, "shared": "native0"},
                ".root_native1": {"native1": True, "shared": "native1"},
                ".root_script0": {"script0": True, "shared": "script0"},
                ".root_script1": {"script1": True, "shared": "script1"},
                ".nested_native0": {"native0": True, "shared": "native0"},
                ".nested_native1": {"native1": True, "shared": "native1"},
                ".nested_script0": {"script0": True, "shared": "script0"},
                ".nested_script1": {"script1": True, "shared": "script1"},
                "layered()=": {
                    "<<$": "root.root_native0",
                    "<<1$": "root.root_native1",
                    "<<^": "% return root.root_script0\n",
                    "<<1^": "% return root.root_script1\n",
                    "nested": {
                        "<<$": "root.nested_native0",
                        "<<1$": "root.nested_native1",
                        "<<^": "% return root.nested_script0\n",
                        "<<1^": "% return root.nested_script1\n",
                    },
                },
                "result": "=$layered()",
            },
            emit_messages=False,
        )
        self.assertEqual(
            result,
            {
                "result": {
                    "native0": True,
                    "native1": True,
                    "script0": True,
                    "script1": True,
                    "shared": "script1",
                    "nested": {
                        "native0": True,
                        "native1": True,
                        "script0": True,
                        "script1": True,
                        "shared": "script1",
                    },
                }
            },
        )


    def test_structural_function_declaration_and_argument_errors(self) -> None:
        invalid = [
            {"scalar(x)=": "x"},
            {"unsuffixed(x)": {"value": 1}},
            {"bad(1x)=": {}},
            {"same(x)=": {}, "same(x)$": "x"},
        ]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(jinest.JinestError):
                jinest.resolve(data, emit_messages=False)

        with self.assertRaisesRegex(jinest.JinestFunctionError, "missing required"):
            jinest.resolve(
                {"build(value)=": {"value$": "value"}, "result": "=$build()"},
                emit_messages=False,
            )
        with self.assertRaisesRegex(jinest.JinestFunctionError, "unknown argument"):
            jinest.resolve(
                {"build(value)=": {"value$": "value"}, "result": "=$build(other=1)"},
                emit_messages=False,
            )

    def test_function_script_local_shadows_argument(self) -> None:
        result = jinest.resolve(
            {
                "rewrite(value)^": "% set value = 9\n% return value\n",
                "result$": "rewrite(5)",
            }
        )
        self.assertEqual(result, {"result": 9})

    def test_functions_are_lazy_and_not_materialized(self) -> None:
        result = jinest.resolve(
            {
                "unused(value)$": "missing.name",
                "namespace": {"only(value)$": "missing.name"},
                "ok": 1,
            }
        )
        self.assertEqual(result, {"namespace": {}, "ok": 1})

    def test_function_collisions_and_invalid_declarations(self) -> None:
        invalid = [
            {"same": 1, "same(x)$": "x"},
            {"same(x)$": "x", "same(x)@": "x"},
            {"same(x)$": "x", "same(y)$": "y"},
            {"bad(x)@@": "x"},
            {"bad(1x)$": "x"},
            {"bad(a, a)$": "a"},
        ]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(jinest.JinestError):
                jinest.resolve(data)

    def test_function_argument_errors_and_recursion_limit(self) -> None:
        with self.assertRaisesRegex(jinest.JinestFunctionError, "missing required"):
            jinest.resolve({"f(a, b)$": "a + b", "result$": "f(1)"})
        with self.assertRaisesRegex(jinest.JinestFunctionError, "unknown argument"):
            jinest.resolve({"f(a)$": "a", "result$": "f(a=1, b=2)"})
        with self.assertRaisesRegex(jinest.JinestFunctionError, "duplicate argument"):
            jinest.resolve({"f(a)$": "a", "result$": "f(1, a=2)"})
        with self.assertRaisesRegex(jinest.JinestFunctionError, "recursion limit"):
            jinest.Resolver(
                {"loop()$": "loop()", "result$": "loop()"},
                function_max_depth=4,
            ).resolve()

    def test_function_attributes_are_sandboxed(self) -> None:
        with self.assertRaisesRegex(jinest.JinestTemplateError, "unsafe"):
            jinest.resolve(
                {"square(x)$": "x * x", "result$": "square.__class__"}
            )


class JinestMessageTests(unittest.TestCase):
    def test_priority_warnings_and_hidden_hint_are_collected_and_printed(self) -> None:
        resolver = jinest.Resolver(
            {
                "name": "literal",
                "name^": "% return missing.value\n",
                "name$": "missing.value",
                "name@": "{{ missing.value }}",
                ".key": "intermediate",
                "key": "public",
            }
        )
        self.assertEqual(len(resolver.messages), 4)
        self.assertTrue(all(message.level == "warning" for message in resolver.messages[:3]))
        self.assertEqual(resolver.messages[3].level, "hint")
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            self.assertEqual(resolver.resolve(), {"name": "literal", "key": "public"})
        rendered = stream.getvalue()
        self.assertIn("warning: Field 'name' suppresses 'name^'", rendered)
        self.assertIn("warning: Field 'name' suppresses 'name$'", rendered)
        self.assertIn("warning: Field 'name' suppresses 'name@'", rendered)
        self.assertIn("hint: Hidden field '.key'", rendered)

    def test_same_declarations_in_different_mappings_are_not_collapsed(self) -> None:
        resolver = jinest.Resolver(
            {
                "left": {"value": 1, "value$": "2"},
                "right": {"value": 3, "value$": "4"},
            },
            emit_messages=False,
        )
        self.assertEqual(resolver.resolve(), {"left": {"value": 1}, "right": {"value": 3}})
        warnings = [message for message in resolver.messages if message.level == "warning"]
        self.assertEqual(len(warnings), 2)

    def test_combinator_declarations_follow_field_priority(self) -> None:
        resolver = jinest.Resolver(
            {
                "value": "winner",
                "value^": "% return 'script'\n",
                "value$": "native",
                "value@": "text",
                "value*": [[1]],
                "value+": [[2]],
                "value%": [[3]],
                "value~": ["join"],
            },
            emit_messages=False,
        )
        self.assertEqual(resolver.resolve(), {"value": "winner"})
        self.assertEqual(len(resolver.messages), 7)
        self.assertTrue(
            all(
                "name > name^ > name$ > name@ > name* > name+ > name% > name~"
                in message.msg
                for message in resolver.messages
            )
        )

    def test_message_output_can_be_disabled(self) -> None:
        resolver = jinest.Resolver(
            {"value": 1, "value$": "2", ".value": 3},
            emit_messages=False,
        )
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            self.assertEqual(resolver.resolve(), {"value": 1})
        self.assertEqual(stream.getvalue(), "")
        self.assertEqual([message.level for message in resolver.messages], ["warning", "hint"])

    def test_warnings_can_be_treated_as_errors(self) -> None:
        resolver = jinest.Resolver(
            {"value": 1, "value$": "2"},
            emit_messages=False,
            treat_warnings_as_errors=True,
        )
        with self.assertRaisesRegex(jinest.JinestWarningError, "Warnings treated as errors"):
            resolver.resolve()
        self.assertEqual(len(resolver.messages), 1)

    def test_debug_adds_message_location(self) -> None:
        resolver = jinest.Resolver(
            {"value": 1, "value$": "2"},
            debug=True,
        )
        self.assertEqual(resolver.messages[0].path, "root")
        self.assertIsNone(resolver.messages[0].file)
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            self.assertEqual(resolver.resolve(), {"value": 1})
        self.assertIn("jinest: warning:", stream.getvalue())
        self.assertIn("  at root\n  in <memory>\n", stream.getvalue())

    def test_cli_reports_unexpected_exceptions_without_a_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            input_path = folder / "input.yml"
            input_path.write_text("value: 1\n", encoding="utf-8")
            process = subprocess.run(
                [
                    sys.executable,
                    str(JINEST_MODULE_PATH),
                    str(input_path),
                    "-o",
                    str(folder),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(process.returncode, 1)
        self.assertEqual(process.stdout, "")
        self.assertTrue(process.stderr.startswith("jinest: "))
        self.assertNotIn("Traceback", process.stderr)

    def test_debug_adds_error_location(self) -> None:
        resolver = jinest.Resolver({"broken$": "missing.value"}, debug=True)
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            with self.assertRaisesRegex(
                jinest.JinestTemplateError,
                "Failed to render",
            ):
                resolver.resolve()
        self.assertIn("jinest: Failed to render", stream.getvalue())
        self.assertIn("  at root['broken$']\n  in <memory>\n", stream.getvalue())

    def test_debug_message_file_location(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yml"
            path.write_text("value: 1\nvalue$: 2\n", encoding="utf-8")
            resolver = jinest.Resolver(
                {"value": 1, "value$": 2},
                source_path=path,
                debug=True,
            )
            self.assertEqual(resolver.messages[0].path, "root")
            self.assertEqual(resolver.messages[0].file, str(path.resolve()))


    @unittest.skipUnless(yaml is not None, "PyYAML is required for YAML tests")
    def test_imported_diagnostics_keep_source_location_and_global_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            library = folder / "library.yaml"
            main = folder / "main.yaml"
            library.write_text("value: literal\nvalue$: '2'\n", encoding="utf-8")
            main.write_text(
                "left$: import('library.yaml')\nright$: import('library.yaml')\n",
                encoding="utf-8",
            )
            resolver = jinest.Resolver(
                {
                    "left$": "import('library.yaml')",
                    "right$": "import('library.yaml')",
                },
                source_path=main,
                emit_messages=False,
                treat_warnings_as_errors=True,
            )

            with self.assertRaises(jinest.JinestWarningError):
                resolver.resolve()

        warnings = [message for message in resolver.messages if message.level == "warning"]
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].path, "root")
        self.assertEqual(warnings[0].file, str(library.resolve()))

    @unittest.skipUnless(yaml is not None, "PyYAML is required for YAML tests")
    def test_debug_imported_error_uses_declaration_location(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            library = folder / "library.yaml"
            main = folder / "main.yaml"
            library.write_text("broken$: missing.value\n", encoding="utf-8")
            main.write_text("instance$: import('library.yaml')\n", encoding="utf-8")
            resolver = jinest.Resolver(
                {"instance$": "import('library.yaml')"},
                source_path=main,
                emit_messages=False,
                debug=True,
            )
            stream = io.StringIO()
            with contextlib.redirect_stderr(stream):
                with self.assertRaises(jinest.JinestTemplateError):
                    resolver.resolve()

        rendered = stream.getvalue()
        self.assertIn("  at root['broken$']\n", rendered)
        self.assertIn(f"  in {library.resolve()}\n", rendered)

class JinestInlineSyntaxTests(unittest.TestCase):
    def test_raw_keys_disable_all_key_parsing(self) -> None:
        resolver = jinest.Resolver(
            {
                "value$`": "literal native-looking key",
                "value@`": "literal text-looking key",
                "value^`": "literal script-looking key",
                "final``": "one final backtick",
                "value$": "40 + 2",
            },
            emit_messages=False,
        )
        self.assertEqual(
            resolver.resolve(),
            {
                "value$": "literal native-looking key",
                "value@": "literal text-looking key",
                "value^": "literal script-looking key",
                "final`": "one final backtick",
                "value": 42,
            },
        )
        self.assertEqual(resolver.messages, [])

    def test_raw_and_dynamic_dot_keys_remain_literal(self) -> None:
        result = jinest.resolve(
            {
                "=$'.dynamic'": 1,
                ".raw`": 2,
                ".hidden": 3,
            },
            emit_messages=False,
        )
        self.assertEqual(result, {".dynamic": 1, ".raw": 2})

    def test_inline_directives_in_mappings_and_arrays(self) -> None:
        result = jinest.resolve(
            {
                "value": 4,
                "native": "=$value * 2",
                "text": "=@value={{ value }}",
                "script": "=^% return {'value': value * 3}\n",
                "escaped": "`=$value",
                "items": [
                    "=$value + 1",
                    "=@item={{ value }}",
                    "=^% return value * 4\n",
                    "`=@item={{ value }}",
                ],
            },
            emit_messages=False,
        )
        self.assertEqual(
            result,
            {
                "value": 4,
                "native": 8,
                "text": "value=4",
                "script": {"value": 12},
                "escaped": "=$value",
                "items": [5, "item=4", 16, "=@item={{ value }}"],
            },
        )

    def test_dynamic_keys_are_literal_evaluated_once_and_checked(self) -> None:
        calls: list[None] = []

        def dynamic_name() -> str:
            calls.append(None)
            return "result$"

        resolver = jinest.Resolver(
            {
                "base": 21,
                "=$dynamic_name()": "=$base * 2",
                "=$'visible@'": "literal final key",
                "=@text-key": "text dynamic key",
                "=^% return 'script-key'\n": "script dynamic key",
            },
            globals={"dynamic_name": dynamic_name},
            emit_messages=False,
        )
        self.assertEqual(
            list(resolver.root),
            ["base", "result$", "visible@", "text-key", "script-key"],
        )
        self.assertEqual(resolver.root["result$"], 42)
        self.assertEqual(
            resolver.resolve(),
            {
                "base": 21,
                "result$": 42,
                "visible@": "literal final key",
                "text-key": "text dynamic key",
                "script-key": "script dynamic key",
            },
        )
        self.assertEqual(len(calls), 1)

        with self.assertRaisesRegex(jinest.JinestError, "expected a string"):
            jinest.resolve({"=$42": "value"}, emit_messages=False)
        with self.assertRaisesRegex(jinest.JinestError, "Duplicate dynamic mapping key"):
            jinest.resolve({"key": 1, "=$'key'": 2}, emit_messages=False)

    def test_declaration_logical_name_collisions_are_rejected(self) -> None:
        cases = [
            {"f": {"<(x)=": {"value$": "x"}}, "f$": "42"},
            {"name": "f", "=$name": 1, "f(x)=": {"value$": "x"}},
            {"name": "f", "=$name": 1, "f[x=[1]]=": []},
            {"f": {"<[x=[1]]=": []}, "f$": "42"},
            {"f": {"<(x)=": {"value$": "x"}}, ".f": 1},
        ]
        for source in cases:
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                    jinest.JinestError,
                    r"Duplicate (?:logical name|dynamic mapping key)",
                ):
                    jinest.resolve(source, emit_messages=False)

        resolver = jinest.Resolver(
            {"nested": cases[0]},
            emit_messages=False,
        )
        for attempt in range(2):
            with self.subTest(cache_retry=attempt):
                with self.assertRaisesRegex(
                    jinest.JinestError,
                    "Duplicate logical name",
                ):
                    resolver.resolve()

    def test_self_wrapper_uses_mapping_protocol_for_proxy_items_field(self) -> None:
        """A user field named items must not shadow parser internals."""
        result = jinest.resolve(
            {
                "items": ["sentinel"],
                "make()=": {"value": 1},
                "array": "=$[root.make()]",
            },
            emit_messages=False,
        )
        self.assertEqual(result["array"], [{"value": 1}])

    def test_self_dollar_and_inline_layers_are_equivalent(self) -> None:
        result = jinest.resolve(
            {
                "x": 2,
                "suffix$": "x + 1",
                "inline": "=$x + 1",
                "wrapped": {"<$": "x + 1"},
            },
            emit_messages=False,
        )
        self.assertEqual(
            result,
            {"x": 2, "suffix": 3, "inline": 3, "wrapped": 3},
        )

    def test_self_text_and_script_wrappers_use_standard_evaluators(self) -> None:
        result = jinest.resolve(
            {
                "value": 2,
                "text_source": "hello",
                "text": {"<@": "{{ text_source }} {{ value }}"},
                "script": {"<^": "% return value + 3\n"},
            },
            emit_messages=False,
        )
        self.assertEqual(
            result,
            {
                "value": 2,
                "text_source": "hello",
                "text": "hello 2",
                "script": 5,
            },
        )

    def test_self_dollar_pipeline_and_type_validation(self) -> None:
        result = jinest.resolve(
            {
                "x": 2,
                "pipeline$": {"<$": '"x + 1"'},
                "nested": {"<$": {"<$": '"x + 1"'}},
            },
            emit_messages=False,
        )
        self.assertEqual(result, {"x": 2, "pipeline": 3, "nested": 3})
        with self.assertRaisesRegex(jinest.JinestError, "requires a string body"):
            jinest.resolve(
                {"x": 2, "bad$": {"<$": "{'nested': x}"}},
                emit_messages=False,
            )

    def test_inline_directives_in_arrays_apply_without_field_mode(self) -> None:
        result = jinest.resolve(
            {
                "value": 2,
                "items": [
                    "=$value + 1",
                    "=@{{ value }}",
                    "=^% return value * 3\n",
                    "`=$value",
                ],
            },
            emit_messages=False,
        )
        self.assertEqual(result, {"value": 2, "items": [3, "2", 6, "=$value"]})

    def test_self_structural_function_and_compose(self) -> None:
        result = jinest.resolve(
            {
                "function_name": "make",
                "=$function_name": {"<(x)=": {"value$": "x"}},
                "function_result$": "make(7).value",
                "items": [1, 2],
                "compose_name": "expanded",
                "=$compose_name": {"<[item=items]=": ["=$item"]},
            },
            emit_messages=False,
        )
        self.assertEqual(
            result,
            {
                "function_name": "make",
                "function_result": 7,
                "items": [1, 2],
                "compose_name": "expanded",
                "expanded": [1, 2],
            },
        )

    def test_self_compose_rebinds_an_array_slot(self) -> None:
        result = jinest.resolve(
            {
                "values": [1, 2],
                "items": [
                    {
                        "<[value=values]=": [
                            {"value$": "value", "where$": "path"}
                        ]
                    }
                ],
            },
            emit_messages=False,
        )
        self.assertEqual(
            result,
            {
                "values": [1, 2],
                "items": [
                    [
                        {"value": 1, "where": "global_root.items[0][0]"},
                        {"value": 2, "where": "global_root.items[0][1]"},
                    ]
                ],
            },
        )

    def test_self_wrapper_requires_exactly_one_mapping_key(self) -> None:
        result = jinest.resolve(
            {"x": 2, "value": {"<$": "x + 1", "other": 4}},
            emit_messages=False,
        )
        self.assertEqual(result, {"x": 2, "value": {"<$": "x + 1", "other": 4}})

    def test_inline_directive_composes_with_field_mode(self) -> None:
        resolver = jinest.Resolver(
            {
                "value": 2,
                "native$": "=@'winner={{ keymode }}'",
                "text@": '=^% return "winner={{ keymode }}"\n',
                "equivalent_old$": "value + 1",
                "equivalent_new": "=$value + 1",
            },
            emit_messages=False,
        )
        self.assertEqual(
            resolver.resolve(),
            {
                "value": 2,
                "native": "winner=@",
                "text": "winner=@",
                "equivalent_old": 3,
                "equivalent_new": 3,
            },
        )
        self.assertEqual(resolver.messages, [])

    def test_inline_directives_in_arrays_are_explicit_item_layers(self) -> None:
        resolver = jinest.Resolver(
            {
                "value": 2,
                "items": [
                    "=$value + 1",
                    "=@{{ value }}",
                    "=^% return value * 3\n",
                    "`=$value",
                ],
            },
            emit_messages=False,
        )
        self.assertEqual(
            resolver.resolve(),
            {"value": 2, "items": [3, "2", 6, "=$value"]},
        )
        self.assertEqual(resolver.messages, [])


class JinestComposeTests(unittest.TestCase):
    def test_nested_structural_compose_preserves_inner_axis_locals(self) -> None:
        """Nested compose frames must retain both outer and inner axes."""
        result = jinest.resolve(
            {
                "values": [1, 2],
                "outer[i=values]=": [
                    {
                        "inner[j=values]=": [
                            {"outer_value$": "i", "inner_value$": "j"}
                        ]
                    }
                ],
            },
            emit_messages=False,
        )
        self.assertEqual(
            result,
            {
                "values": [1, 2],
                "outer": [
                    {
                        "inner": [
                            {"outer_value": 1, "inner_value": 1},
                            {"outer_value": 1, "inner_value": 2},
                        ]
                    },
                    {
                        "inner": [
                            {"outer_value": 2, "inner_value": 1},
                            {"outer_value": 2, "inner_value": 2},
                        ]
                    },
                ],
            },
        )

    def test_structural_compose_axis_order_mapping_and_text(self) -> None:
        result = jinest.resolve(
            {
                "versions": ["3.10", "3.11"],
                "dirs": ["bin", "lib"],
                "prefix": "run",
                "items[v=versions, d=dirs]=": [
                    "=@{{ prefix }}-{{ v }}-{{ d }}",
                    "=$v ~ ':' ~ d",
                ],
                "by_version[v=versions]=": {"=$'py' ~ v": "=$v"},
                "text[v=versions, d=dirs]@": "{{ v }}:{{ d }};",
            },
            emit_messages=False,
        )
        self.assertEqual(
            result,
            {
                "versions": ["3.10", "3.11"],
                "dirs": ["bin", "lib"],
                "prefix": "run",
                "items": [
                    "run-3.10-bin", "3.10:bin",
                    "run-3.10-lib", "3.10:lib",
                    "run-3.11-bin", "3.11:bin",
                    "run-3.11-lib", "3.11:lib",
                ],
                "by_version": {"py3.10": "3.10", "py3.11": "3.11"},
                "text": "3.10:bin;3.10:lib;3.11:bin;3.11:lib;",
            },
        )

    def test_structural_compose_rebinds_each_body_and_keeps_axis_frame(self) -> None:
        result = jinest.resolve(
            {
                "values": [10, 20],
                "items[value=values]=": [
                    {
                        "value$": "value",
                        "path$": "path",
                        "nested": {"path$": "path", "value$": "value"},
                    }
                ],
            },
            emit_messages=False,
        )
        self.assertEqual(
            result["items"],
            [
                {
                    "value": 10,
                    "path": "global_root.items[0]",
                    "nested": {
                        "path": "global_root.items[0].nested",
                        "value": 10,
                    },
                },
                {
                    "value": 20,
                    "path": "global_root.items[1]",
                    "nested": {
                        "path": "global_root.items[1].nested",
                        "value": 20,
                    },
                },
            ],
        )

    def test_compose_axes_follow_jinja_iteration_protocol(self) -> None:
        result = jinest.resolve(
            {
                "from_iterator[value=iterator]=": ["=$value"],
                "from_mapping[key=mapping]=": ["=$key"],
            },
            globals={
                "iterator": iter(("first", "second")),
                "mapping": {"left": 1, "right": 2},
            },
            emit_messages=False,
        )
        self.assertEqual(
            result,
            {
                "from_iterator": ["first", "second"],
                "from_mapping": ["left", "right"],
            },
        )

    def test_compose_iteration_metadata(self) -> None:
        result = jinest.resolve(
            {
                "versions": ["a", "b"],
                "dirs": ["x", "y"],
                "items[v=versions, d=dirs]=": [
                    {
                        "value@": "{{ v }}/{{ d }}",
                        "axis_index$": "axis.v.index",
                        "axis_index0$": "axis.d.index0",
                        "axis_length$": "axis.v.length",
                        "axis_first$": "axis.d.first",
                        "axis_last$": "axis.v.last",
                        "product_index$": "axes.index",
                        "product_index0$": "axes.index0",
                        "product_length$": "axes.length",
                        "product_first$": "axes.first",
                        "product_last$": "axes.last",
                        "nested": {
                            "position@": "{{ axis.v.index }}/{{ axes.index }}"
                        },
                    }
                ],
                "by_key[v=versions, d=dirs]=": {
                    "=$v ~ '-' ~ d": "=$axes.index ~ ':' ~ axis.v.index"
                },
                "text[v=versions, d=dirs]@": (
                    "{{ v }}{{ d }}:"
                    "{{ axis.v.index }}/{{ axis.d.index }}:"
                    "{{ axes.index }}/{{ axes.index0 }};"
                ),
                "empty": [],
                "empty_items[v=empty]=": ["=$axes.length"],
                "empty_text[v=empty]@": "{{ axes.length }}",
            },
            emit_messages=False,
        )
        self.assertEqual(
            result["items"],
            [
                {
                    "value": "a/x",
                    "axis_index": 1,
                    "axis_index0": 0,
                    "axis_length": 2,
                    "axis_first": True,
                    "axis_last": False,
                    "product_index": 1,
                    "product_index0": 0,
                    "product_length": 4,
                    "product_first": True,
                    "product_last": False,
                    "nested": {"position": "1/1"},
                },
                {
                    "value": "a/y",
                    "axis_index": 1,
                    "axis_index0": 1,
                    "axis_length": 2,
                    "axis_first": False,
                    "axis_last": False,
                    "product_index": 2,
                    "product_index0": 1,
                    "product_length": 4,
                    "product_first": False,
                    "product_last": False,
                    "nested": {"position": "1/2"},
                },
                {
                    "value": "b/x",
                    "axis_index": 2,
                    "axis_index0": 0,
                    "axis_length": 2,
                    "axis_first": True,
                    "axis_last": True,
                    "product_index": 3,
                    "product_index0": 2,
                    "product_length": 4,
                    "product_first": False,
                    "product_last": False,
                    "nested": {"position": "2/3"},
                },
                {
                    "value": "b/y",
                    "axis_index": 2,
                    "axis_index0": 1,
                    "axis_length": 2,
                    "axis_first": False,
                    "axis_last": True,
                    "product_index": 4,
                    "product_index0": 3,
                    "product_length": 4,
                    "product_first": False,
                    "product_last": True,
                    "nested": {"position": "2/4"},
                },
            ],
        )
        self.assertEqual(
            result["by_key"],
            {"a-x": "1:1", "a-y": "2:1", "b-x": "3:2", "b-y": "4:2"},
        )
        self.assertEqual(
            result["text"],
            "ax:1/1:1/0;ay:1/2:2/1;bx:2/1:3/2;by:2/2:4/3;",
        )
        self.assertEqual(result["empty_items"], [])
        self.assertEqual(result["empty_text"], "")

    def test_compose_axis_names_can_be_index_and_length(self) -> None:
        result = jinest.resolve(
            {
                "indexes": [10, 20],
                "lengths": ["short", "long"],
                "items[index=indexes, length=lengths]=": [
                    {
                        "value$": "index",
                        "local_index$": "axis.index.index",
                        "local_index0$": "axis.index.index0",
                        "local_length$": "axis.length.length",
                        "global_index$": "axes.index",
                        "global_index0$": "axes.index0",
                    }
                ],
            },
            emit_messages=False,
        )
        self.assertEqual(
            result["items"],
            [
                {
                    "value": 10,
                    "local_index": 1,
                    "local_index0": 0,
                    "local_length": 2,
                    "global_index": 1,
                    "global_index0": 0,
                },
                {
                    "value": 10,
                    "local_index": 1,
                    "local_index0": 0,
                    "local_length": 2,
                    "global_index": 2,
                    "global_index0": 1,
                },
                {
                    "value": 20,
                    "local_index": 2,
                    "local_index0": 1,
                    "local_length": 2,
                    "global_index": 3,
                    "global_index0": 2,
                },
                {
                    "value": 20,
                    "local_index": 2,
                    "local_index0": 1,
                    "local_length": 2,
                    "global_index": 4,
                    "global_index0": 3,
                },
            ],
        )

    def test_compose_rejects_reserved_axis_names(self) -> None:
        for name in ("axis", "axes"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(jinest.JinestError, "is reserved"):
                    jinest.Resolver({f"bad[{name}=items]=": []})

    def test_compose_errors(self) -> None:
        with self.assertRaisesRegex(jinest.JinestError, "must resolve to an iterable"):
            jinest.resolve({"bad[value=42]=": []}, emit_messages=False)
        with self.assertRaisesRegex(jinest.JinestError, "Duplicate dynamic mapping key"):
            jinest.resolve(
                {"values": [1, 2], "bad[value=values]=": {"same": "=$value"}},
                emit_messages=False,
            )
        with self.assertRaisesRegex(jinest.JinestError, "must have a mapping or array body"):
            jinest.Resolver({"bad[value=[1]]=": "not structural"})
        with self.assertRaisesRegex(jinest.JinestError, "must have a string body"):
            jinest.Resolver({"bad[value=[1]]@": []})
        with self.assertRaisesRegex(jinest.JinestError, "must use name=source syntax"):
            jinest.Resolver({"bad[value]=": []})
        with self.assertRaisesRegex(jinest.JinestError, "Type annotations are unsupported"):
            jinest.Resolver({"bad[value: int=[1]]=": []})


class JinestLayerTests(unittest.TestCase):
    def test_legacy_merge_order_is_rejected(self) -> None:
        for key in ("<<1!$", "<<1![]"):
            with self.subTest(key=key), self.assertRaisesRegex(
                jinest.JinestError,
                r"Invalid merge declaration",
            ):
                jinest.resolve({"target": {key: {"value": 1}}}, emit_messages=False)

    def test_structural_function_layer_preserves_full_frame_metadata(self) -> None:
        """Merge layers must preserve the same origin as ordinary attachments."""
        result = jinest.resolve(
            {
                "make(x)=": {
                    "value$": "x",
                    "origin_where@": "{{ origin.path }}",
                    "nested": {"origin_where@": "{{ origin.path }}"},
                },
                "attached": "=$root.make(1)",
                "native_layer": {"<<$": "root.make(2)"},
                "script_layer": {"<<^": "% return root.make(3)\n"},
                "multi_layer": {"<<[]": ["=$root.make(4)"]},
            },
            emit_messages=False,
        )
        for name, value in (
            ("attached", 1),
            ("native_layer", 2),
            ("script_layer", 3),
            ("multi_layer", 4),
        ):
            with self.subTest(name=name):
                self.assertEqual(result[name]["value"], value)
                self.assertEqual(result[name]["origin_where"], "root")
                self.assertEqual(result[name]["nested"]["origin_where"], "root")

    def test_layer_order_defaults_local_overrides(self) -> None:
        data = {
            "defaults1": {"rank": "d1", "d1": True},
            "defaults2": {"rank": "d2", "d2": True},
            "overrides1": {"rank": "o1", "o1": True},
            "overrides2": {"rank": "o2", "o2": True},
            "example": {
                # Deliberately scrambled source order. Numeric order controls
                # each family independently.
                "<<!2$": "root.overrides2",
                "<<2$": "root.defaults2",
                "rank": "local",
                "<<!1$": "root.overrides1",
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

    def test_multi_source_layers_expand_inline_items_and_orders(self) -> None:
        result = jinest.resolve(
            {
                "first": {"from_first": True, "rank": "first"},
                "second": {"from_second": True, "rank": "second"},
                "single": {"rank": "single"},
                "target": {
                    "<<2[]": [
                        "=$root.first",
                        "=^% return root.second",
                        {"literal": True},
                    ],
                    "<<2$": "root.single",
                    "rank": "local",
                    "<<!1[]": ["=$root.first"],
                },
            }
        )
        self.assertEqual(
            result["target"],
            {
                "from_first": True,
                "from_second": True,
                "literal": True,
                "rank": "first",
            },
        )

    def test_multi_source_layer_all_declared_forms(self) -> None:
        result = jinest.resolve(
            {
                "default": {"value": "default"},
                "override": {"value": "override"},
                "numbered": {"numbered": True},
                "plain": {
                    "<<[]": ["=$root.default"],
                    "<<![]": ["=$root.override"],
                },
                "numbered_target": {"<<1[]": ["=$root.numbered"]},
            }
        )
        self.assertEqual(result["plain"], {"value": "override"})
        self.assertEqual(result["numbered_target"], {"numbered": True})

    def test_multi_source_layer_single_has_priority_at_same_order(self) -> None:
        result = jinest.resolve(
            {
                "array_layer": {"value": "array"},
                "single_layer": {"value": "single"},
                "target": {
                    "<<7[]": ["=$root.array_layer"],
                    "<<7$": "root.single_layer",
                },
            }
        )
        self.assertEqual(result["target"], {"value": "single"})

    def test_multi_source_layer_items_are_lazy_and_can_depend_on_prior_layers(self) -> None:
        resolver = jinest.Resolver(
            {
                "target": {
                    # Binding this mapping must not evaluate list items.
                    "<<[]": ["=$missing_layer", {"base": 1}],
                },
            }
        )
        # Merely binding the surrounding mapping does not evaluate an array
        # item; item evaluation begins only when merge lookup needs it.
        self.assertEqual(str(resolver.root["target"].path), "global_root.target")

        result = jinest.resolve(
            {
                "target": {
                    "<<[]": [{"base": 1}],
                    # Normalization must expose the preceding array layer to
                    # this override item, just like <<$ / <<!$.
                    "<<![]": ["=${'value': base}"],
                }
            }
        )
        self.assertEqual(result["target"], {"base": 1, "value": 1})

    def test_structural_function_layer_results_keep_argument_locals(self) -> None:
        data = {
            "make(x)=": {"value$": "x"},
            "from_single": {"<<$": "root.make(3)"},
            "from_script": {"<<^": "% return root.make(4)"},
            "from_many": {"<<[]": ["=$root.make(5)"]},
            "nested(x)=": {
                "<<[]": [{"base$": "x"}],
                "value$": "base",
            },
            "from_nested": {"<<[]": ["=$root.nested(6)"]},
        }
        result = jinest.resolve(data)
        self.assertEqual(result["from_single"], {"value": 3})
        self.assertEqual(result["from_script"], {"value": 4})
        self.assertEqual(result["from_many"], {"value": 5})
        self.assertEqual(result["from_nested"], {"base": 6, "value": 6})

    def test_multi_source_layers_are_normalized_but_fields_stay_lazy(self) -> None:
        resolver = jinest.Resolver(
            {
                "base": {"safe": 1, "broken$": "missing_name"},
                "target": {"<<[]": ["=$root.base"]},
            }
        )
        target = resolver.root["target"]
        self.assertEqual(target["safe"], 1)
        with self.assertRaises(jinest.JinestTemplateError):
            _ = target["broken"]

    def test_multi_source_layer_requires_list_of_mappings(self) -> None:
        with self.assertRaisesRegex(jinest.JinestMergeError, "expected a list"):
            jinest.resolve({"target": {"<<[]": {"x": 1}}})
        with self.assertRaisesRegex(
            jinest.JinestMergeError, r"target\['<<\[\]'\]\[0\].*expected a mapping"
        ):
            jinest.resolve({"target": {"<<[]": ["=$42"]}})


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

    def test_source_access_does_not_change_destination_binding(self) -> None:
        resolver = jinest.Resolver(self._prototype_data())
        # Access the declaration at its source location first.
        self.assertEqual(resolver.root["class"].prototype.A, 1)
        self.assertEqual(resolver.root["class"].prototype.where, "global_root.class.prototype")
        # The destination must still resolve against its own context.
        self.assertEqual(resolver.root.inherited.instance.A, 2)
        self.assertEqual(
            resolver.root.inherited.instance.where,
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
    def test_legacy_mode_typed_arrays_are_rejected(self) -> None:
        """Evaluator suffixes remain scalar-only; item evaluation is explicit."""
        for source in (
            {"items$": ["1"]},
            {"items@": ["1"]},
            {"items^": ["% return 1"]},
        ):
            with self.subTest(source=source):
                with self.assertRaisesRegex(
                    jinest.JinestTemplateError,
                    "requires a string body",
                ):
                    jinest.resolve(source, emit_messages=False)

    def test_native_and_text_array_items(self) -> None:
        result = jinest.resolve(
            {
                "var1": 7,
                "var2": 9,
                "native_array": [
                    "=$var1",
                    "=$root.var2",
                    "=$1",
                    "=$true",
                    5,
                    "none",
                    "=$path",
                ],
                "text_array": [
                    "=@{{ var1 }}",
                    "=@v={{ root.var2 }}",
                    "=@{{ 1 }}",
                    "=@{{ true }}",
                    "=@{{ path }}",
                ],
            }
        )
        self.assertEqual(
            result["native_array"],
            [7, 9, 1, True, 5, "none", "global_root.native_array[6]"],
        )
        self.assertEqual(
            result["text_array"],
            ["7", "v=9", "1", "True", "global_root.text_array[4]"],
        )

    def test_array_is_lazy_per_item(self) -> None:
        resolver = jinest.Resolver(
            {
                "items": [
                    "=$40 + 2",
                    "=$missing.deep.value",
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
        resolver = jinest.Resolver({"items": ["=$1", "=$2", "=$3"]})
        self.assertEqual(resolver.root.items[-1], 3)
        self.assertEqual(resolver.root.items[0:2], [1, 2])
        with self.assertRaises(IndexError):
            _ = resolver.root.items[10]

    def test_array_combinator_modes_and_edge_cases(self) -> None:
        result = jinest.resolve(
            {
                "flatten+": [[1, 2], [3], [4, 5]],
                "tuple_flatten+": ((1, 2), (3,)),
                "nested_flatten+": [[[1]], [[2]]],
                "join~": ["one", "-", "two"],
                "empty_join~": [],
                "product*": [[1, 2], ["a", "b"]],
                "empty_product*": [],
                "zip%": [[1, 2], ["a", "b"]],
                "empty_zip%": [],
                "empty_axis_product*": [[1], []],
                "zero_axis_product*": [],
            },
            emit_messages=False,
        )
        self.assertEqual(result["flatten"], [1, 2, 3, 4, 5])
        self.assertEqual(result["tuple_flatten"], [1, 2, 3])
        self.assertEqual(result["nested_flatten"], [[1], [2]])
        self.assertEqual(result["join"], "one-two")
        self.assertEqual(result["empty_join"], "")
        self.assertEqual(
            result["product"],
            [[1, "a"], [1, "b"], [2, "a"], [2, "b"]],
        )
        self.assertEqual(result["empty_product"], [[]])
        self.assertEqual(result["zip"], [[1, "a"], [2, "b"]])
        self.assertEqual(result["empty_zip"], [])
        self.assertEqual(result["empty_axis_product"], [])
        self.assertEqual(result["zero_axis_product"], [[]])

    def test_array_combinators_compose_with_inner_native_layer(self) -> None:
        result = jinest.resolve(
            {
                "axes": [["left", "right"], ["up", "down"]],
                "product*": "=$axes",
            },
            emit_messages=False,
        )
        self.assertEqual(
            result["product"],
            [
                ["left", "up"],
                ["left", "down"],
                ["right", "up"],
                ["right", "down"],
            ],
        )

    def test_array_combinators_rebind_structural_nodes(self) -> None:
        result = jinest.resolve(
            {
                "node": {"value$": 42, "path$": "path"},
                "groups$": "[[root.node], [root.node]]",
                "flattened+": "=$groups",
            },
            emit_messages=False,
        )
        self.assertEqual(
            result["flattened"],
            [
                {"value": 42, "path": "global_root.flattened[0]"},
                {"value": 42, "path": "global_root.flattened[1]"},
            ],
        )

    def test_array_combinators_reject_invalid_types(self) -> None:
        invalid = [
            {"flatten+": [1, 2]},
            {"flatten+": [[1], 2]},
            {"join~": ["one", 2]},
            {"product*": [1, [2]]},
            {"zip%": [[1], [2, 3]]},
            {"flatten+": {"not": [1, 2]}},
        ]
        for data in invalid:
            with self.subTest(data=data), self.assertRaisesRegex(
                jinest.JinestError,
                "Array suffix",
            ):
                jinest.resolve(data, emit_messages=False)


@unittest.skipUnless(yaml is not None, "PyYAML is required for YAML tests")
class JinestFormatAndImportTests(unittest.TestCase):
    @staticmethod
    def _serialize_public_value(value: Any, output_format: str) -> str:
        """Return a Python value through the documented resolver/serializer API."""
        return jinest.resolve_text(
            json.dumps({"value$": "make_value()"}),
            format="json",
            output_format=output_format,
            globals={"make_value": lambda: value},
            emit_messages=False,
        )

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
        self.assertIn('"payload": "\\u0000ÿA"', rendered)
        self.assertEqual(
            json.loads(rendered),
            {"date": "2026-08-02", "payload": "\x00ÿA"},
        )

        root_bytes = jinest.resolve_text(
            "!!binary AP9B\n",
            format="yaml",
            output_format="json",
        )
        self.assertEqual(root_bytes, '"\\u0000ÿA"')

    def test_json_bytes_use_lossless_latin1_strings(self) -> None:
        payload = bytes(range(256))
        rendered = self._serialize_public_value(
            {
                "bytes": payload,
                "bytearray": bytearray(payload),
                "mapping": {payload: "value"},
            },
            "json",
        )
        decoded = json.loads(rendered)["value"]
        self.assertEqual(decoded["bytes"].encode("latin-1"), payload)
        self.assertEqual(decoded["bytearray"].encode("latin-1"), payload)
        [decoded_key] = decoded["mapping"]
        self.assertEqual(decoded_key.encode("latin-1"), payload)

    def test_yaml_serializes_time_and_bytearray(self) -> None:
        rendered = self._serialize_public_value(
            {"clock": time(12, 30, 45), "buffer": bytearray(b"\x00A")},
            "yaml",
        )
        self.assertEqual(
            yaml.safe_load(rendered)["value"],
            {"clock": "12:30:45", "buffer": b"\x00A"},
        )

    def test_invalid_yaml_is_a_jinest_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.yml"
            path.write_text("broken: [", encoding="utf-8")
            with self.assertRaisesRegex(jinest.JinestError, "Invalid YAML input"):
                jinest.resolve_file(path)

    def test_resolve_text_wraps_invalid_json_and_yaml(self) -> None:
        cases = [
            ("json", "{", "Invalid JSON input"),
            ("yaml", "broken: [", "Invalid YAML input"),
        ]
        for format, text, message in cases:
            with self.subTest(format=format):
                with self.assertRaisesRegex(jinest.JinestError, message):
                    jinest.resolve_text(text, format=format)

    def test_json_mapping_key_normalization_is_lossless(self) -> None:
        rendered = self._serialize_public_value(
            {date(2026, 8, 2): 1, b"A": 2},
            "json",
        )
        self.assertEqual(
            json.loads(rendered)["value"],
            {"2026-08-02": 1, "A": 2},
        )

        collisions = [
            {date(2026, 8, 2): 1, "2026-08-02": 2},
            {b"A": 1, "A": 2},
            {1: 1, "1": 2},
            {None: 1, "null": 2},
        ]
        for value in collisions:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    jinest.JinestError,
                    "Duplicate JSON object key after normalization",
                ):
                    self._serialize_public_value(value, "json")

        with self.assertRaisesRegex(
            jinest.JinestError,
            "Unsupported mapping key",
        ):
            self._serialize_public_value({(1, 2): 3}, "json")

    def test_pathref_mapping_keys_materialize_to_strings(self) -> None:
        result = jinest.resolve(
            {"result^": "% return {path: 1}\n"},
            emit_messages=False,
        )
        self.assertEqual(result, {"result": {"global_root": 1}})
        self.assertEqual(json.loads(json.dumps(result)), result)

        with self.assertRaisesRegex(
            jinest.JinestError,
            "Duplicate mapping key after materialization",
        ):
            jinest.resolve(
                {"result^": "% return {path: 1, (path|string): 2}\n"},
                emit_messages=False,
            )

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

    def test_repeated_import_rebinds_each_destination_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "library.yaml").write_text(
                "source_path$: origin.path\n"
                "destination_path$: path\n",
                encoding="utf-8",
            )
            (folder / "main.yaml").write_text(
                "left$: import('library.yaml')\n"
                "right$: import('library.yaml')\n",
                encoding="utf-8",
            )

            result = json.loads(
                jinest.resolve_file(folder / "main.yaml", output_format="json")
            )

        self.assertEqual(
            result,
            {
                "left": {
                    "source_path": "root",
                    "destination_path": "global_root.left",
                },
                "right": {
                    "source_path": "root",
                    "destination_path": "global_root.right",
                },
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

    def test_import_cycle_uses_the_current_import_branch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "a.json").write_text(
                json.dumps({"c$": 'import_json("c.json")'}),
                encoding="utf-8",
            )
            (folder / "x.json").write_text(
                json.dumps({"c$": 'import_json("c.json")'}),
                encoding="utf-8",
            )
            (folder / "c.json").write_text(
                json.dumps({"a$": 'import_json("a.json")'}),
                encoding="utf-8",
            )

            result = jinest.Resolver(
                {
                    "first$": "import_json('a.json')",
                    "second$": "import_json('x.json')",
                },
                base_dir=folder,
                emit_messages=False,
            ).resolve()

        self.assertEqual(
            result,
            {
                "first": {"c": {"a": None}},
                "second": {"c": {"a": {"c": None}}},
            },
        )

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

    def test_structural_function_preserves_returned_import_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            library_dir = folder / "lib"
            library_dir.mkdir()
            library = library_dir / "library.json"
            library.write_text(
                json.dumps(
                    {
                        "prototype": {
                            "root_file$": "source_file(root)",
                            "origin_path$": "origin.path",
                            "source_path$": "context.source_path",
                            "nested$": "import_json('nested.json')",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (library_dir / "nested.json").write_text(
                '{"value": "library-relative"}', encoding="utf-8"
            )
            (folder / "nested.json").write_text(
                '{"value": "wrong-main-relative"}', encoding="utf-8"
            )

            result = jinest.resolve(
                {
                    "make()=": {
                        "node$": "import_json('lib/library.json').prototype"
                    },
                    "result$": "make()",
                },
                base_dir=folder,
                source_path=folder / "main.json",
                emit_messages=False,
            )

            self.assertEqual(
                result,
                {
                    "result": {
                        "node": {
                            "root_file": str(library.resolve()),
                            "origin_path": "root.prototype",
                            "source_path": "root.prototype",
                            "nested": {"value": "library-relative"},
                        }
                    }
                },
            )

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
                "values": [
                    "=^% return x + 1\n",
                    "=^% set y = x * 2\n% return y\n",
                    "=^% return 3\n",
                    "=^% return none\n",
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
                    "<<!1^": "% return root.overrides\n",
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
                "<<!2$": "root.o2",
                "<<!1^": "% return root.o1\n",
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


class JinestBindingAndCycleContractTests(unittest.TestCase):
    """Public contracts for binding identity and finite cycle handling."""

    def test_one_physical_mapping_has_a_source_view_per_source_path(self) -> None:
        shared = {"origin_path$": "origin.path", "destination_path$": "path"}

        result = jinest.resolve({"left": shared, "right": shared}, emit_messages=False)

        self.assertEqual(
            result,
            {
                "left": {
                    "origin_path": "root.left",
                    "destination_path": "global_root.left",
                },
                "right": {
                    "origin_path": "root.right",
                    "destination_path": "global_root.right",
                },
            },
        )

    def test_source_identity_and_destination_binding_are_distinct(self) -> None:
        result = jinest.resolve(
            {
                "prototype": {
                    "origin_path$": "origin.path",
                    "destination_path$": "path",
                },
                "target$": "root.prototype",
            },
            emit_messages=False,
        )

        self.assertEqual(
            result,
            {
                "prototype": {
                    "origin_path": "root.prototype",
                    "destination_path": "global_root.prototype",
                },
                "target": {
                    "origin_path": "root.prototype",
                    "destination_path": "global_root.target",
                },
            },
        )

    def test_repeated_attachment_has_independent_binding_caches(self) -> None:
        calls: list[None] = []

        def tick() -> int:
            calls.append(None)
            return len(calls)

        result = jinest.resolve(
            {
                ".prototype": {"path$": "path", "sequence$": "tick()"},
                "left$": "root.prototype",
                "right$": "root.prototype",
            },
            globals={"tick": tick},
            emit_messages=False,
        )

        self.assertEqual(
            result,
            {
                "left": {"path": "global_root.left", "sequence": 1},
                "right": {"path": "global_root.right", "sequence": 2},
            },
        )
        self.assertEqual(len(calls), 2)

    def test_shared_templates_keep_modes_and_destination_context_isolated(self) -> None:
        """Compiled Jinja artifacts are reusable but never capture a binding."""
        shared = {
            "native$": "_.marker",
            "text@": "{{ _.marker }}",
            "script^": "% return _.marker\n",
            "native_literal$": "1",
            "text_literal@": "1",
            "script_literal^": "1",
            "origin_path$": "origin.path",
            "destination_path$": "path",
        }

        result = jinest.resolve(
            {
                "left": {"marker": "left", "node": shared},
                "right": {"marker": "right", "node": shared},
            },
            emit_messages=False,
        )

        self.assertEqual(
            result,
            {
                "left": {
                    "marker": "left",
                    "node": {
                        "native": "left",
                        "text": "left",
                        "script": "left",
                        "native_literal": 1,
                        "text_literal": "1",
                        "script_literal": None,
                        "origin_path": "root.left.node",
                        "destination_path": "global_root.left.node",
                    },
                },
                "right": {
                    "marker": "right",
                    "node": {
                        "native": "right",
                        "text": "right",
                        "script": "right",
                        "native_literal": 1,
                        "text_literal": "1",
                        "script_literal": None,
                        "origin_path": "root.right.node",
                        "destination_path": "global_root.right.node",
                    },
                },
            },
        )

    def test_shared_dynamic_keys_are_built_per_destination_binding(self) -> None:
        shared = {"=$_.key": "value", "where$": "path"}

        result = jinest.resolve(
            {
                "left": {"key": "left_key", "node": shared},
                "right": {"key": "right_key", "node": shared},
            },
            emit_messages=False,
        )

        self.assertEqual(
            result,
            {
                "left": {
                    "key": "left_key",
                    "node": {
                        "left_key": "value",
                        "where": "global_root.left.node",
                    },
                },
                "right": {
                    "key": "right_key",
                    "node": {
                        "right_key": "value",
                        "where": "global_root.right.node",
                    },
                },
            },
        )

    def test_runtime_layer_mapping_does_not_reuse_a_cycle_sentinel(self) -> None:
        self.assertEqual(
            jinest.resolve(
                {"target": {"<<$": "make()"}},
                globals={"make": lambda: {"value": 43}},
                emit_messages=False,
            ),
            {"target": {"value": 43}},
        )

    def test_nested_structural_function_calls_are_finite(self) -> None:
        self.assertEqual(
            jinest.resolve(
                {
                    "box(x)=": {"value$": "x"},
                    "result$": "box(box(1))",
                },
                emit_messages=False,
            ),
            {"result": {"value": {"value": 1}}},
        )

    def test_schema_diagnostics_distinguish_shared_source_occurrences(self) -> None:
        shared = {"value": 1, "value$": "2"}
        resolver = jinest.Resolver(
            {"left": shared, "right": shared}, emit_messages=False
        )

        resolver.resolve()

        self.assertEqual(
            [(message.level, message.path) for message in resolver.messages],
            [("warning", "root.left"), ("warning", "root.right")],
        )

    def test_import_diamond_reuses_one_document_and_one_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "common.json").write_text(
                '{"value": 1, "value$": "2"}', encoding="utf-8"
            )
            (folder / "left.json").write_text(
                '{"common$": "import_json(\'common.json\')"}', encoding="utf-8"
            )
            (folder / "right.json").write_text(
                '{"common$": "import_json(\'common.json\')"}', encoding="utf-8"
            )
            resolver = jinest.Resolver(
                {
                    "left$": "import_json('left.json')",
                    "right$": "import_json('right.json')",
                },
                base_dir=folder,
                emit_messages=False,
            )
            self.assertEqual(
                resolver.resolve(),
                {"left": {"common": {"value": 1}}, "right": {"common": {"value": 1}}},
            )

        warnings = [message for message in resolver.messages if message.level == "warning"]
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].file, str(folder / "common.json"))

    def test_physical_python_container_cycle_is_an_error(self) -> None:
        source: dict[str, Any] = {}
        source["self"] = source

        with self.assertRaisesRegex(jinest.JinestError, "Cyclic container reference"):
            jinest.resolve(source, emit_messages=False)

    def test_field_and_merge_cycle_contracts(self) -> None:
        self.assertEqual(
            jinest.resolve({"a$": "b", "b$": "a"}, emit_messages=False),
            {"a": None, "b": None},
        )
        self.assertEqual(
            jinest.resolve(
                {"target": {"<<$": "root.target", "retained": True}},
                emit_messages=False,
            ),
            {"target": {"retained": True}},
        )

    def test_import_cycle_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "a.json").write_text(
                '{"other$": "import(\'b.json\')"}', encoding="utf-8"
            )
            (folder / "b.json").write_text(
                '{"back$": "import(\'a.json\')"}', encoding="utf-8"
            )

            rendered = jinest.resolve_file(
                folder / "a.json", output_format="json", emit_messages=False
            )

        self.assertEqual(json.loads(rendered), {"other": {"back": None}})

    def test_scalar_function_recursion_obeys_the_depth_limit(self) -> None:
        resolver = jinest.Resolver(
            {"loop()$": "loop()", "result$": "loop()"},
            function_max_depth=4,
            emit_messages=False,
        )

        with self.assertRaisesRegex(jinest.JinestFunctionError, "recursion limit"):
            resolver.resolve()

    def test_recursive_rebinding_through_an_ancestor_fails_finitely(self) -> None:
        with self.assertRaisesRegex(jinest.JinestError, "Cyclic container reference"):
            jinest.resolve({"node": {"again$": "context"}}, emit_messages=False)

    @unittest.skipIf(sys.flags.optimize, "the optimized child process is tested once")
    def test_public_resolution_works_under_python_optimized_mode(self) -> None:
        script = "\n".join(
            [
                "import importlib.util",
                "import sys",
                f"module_path = {str(JINEST_MODULE_PATH)!r}",
                "spec = importlib.util.spec_from_file_location('jinest_optimized_smoke', module_path)",
                "module = importlib.util.module_from_spec(spec)",
                "sys.modules[spec.name] = module",
                "spec.loader.exec_module(module)",
                "result = module.resolve({'value$': '40 + 2'}, emit_messages=False)",
                "if result != {'value': 42}: raise SystemExit(f'unexpected result: {result!r}')",
            ]
        )
        completed = subprocess.run(
            [sys.executable, "-O", "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    @unittest.skipIf(sys.flags.optimize, "the optimized child process is tested once")
    def test_self_test_checks_are_active_under_python_optimized_mode(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-O", str(JINEST_MODULE_PATH), "--self-test"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Jinest self-test: OK", completed.stdout)

        source = JINEST_MODULE_PATH.read_text(encoding="utf-8")
        marker = '(resolver.root.example.rank, "o2", "override priority")'
        self.assertIn(marker, source)
        broken_source = source.replace(
            marker,
            '(resolver.root.example.rank, "deliberately-wrong", "override priority")',
            1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            broken_module = Path(temp_dir) / "jinest.py"
            broken_module.write_text(broken_source, encoding="utf-8")
            failed = subprocess.run(
                [sys.executable, "-O", str(broken_module), "--self-test"],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(failed.returncode, 0)
        self.assertTrue(failed.stderr.startswith("jinest: Jinest self-test failed"))
        self.assertIn("Jinest self-test failed", failed.stderr)
        self.assertNotIn("Traceback", failed.stderr)

    def test_runtime_jinja_meets_the_minimum_requirement(self) -> None:
        import jinja2

        version = tuple(int(part) for part in jinja2.__version__.split(".")[:2])
        self.assertGreaterEqual(version, (3, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
