"""Jinest — lazy structured Jinja resolver for Python, JSON and YAML.

Author: Fedir Khodchenko
License: MIT

Syntax
======

* ``name@`` — render a full Jinja text template.
* ``name$`` — evaluate one native Jinja expression.
* ``name^`` — execute a multiline Jinja script with ``%`` line statements and
  ``return`` a native value.
* ``name(args)$``, ``name(args)@``, and ``name(args)^`` — declare lazy safe
  template functions in native, text, or script mode.
* ``.name`` — a hidden field, available to Jinja as ``name`` but omitted from
  the final resolved mapping.
* A key ending in a backtick is a raw literal key; all Jinest key parsing is
  disabled and the marker is removed.
* ``=$expr``, ``=@text``, and ``=^script`` — inline scalar directives for
  native expressions, text templates, and scripts; they also form dynamic
  mapping keys when used as keys.
* Local priority is ``name`` > ``name^`` > ``name$`` > ``name@``.
* ``<<$`` / ``<<N$`` and ``<<^`` / ``<<N^`` add lazy default layers.
* ``<<!$`` / ``<<N!$`` and ``<<!^`` / ``<<N!^`` add lazy override layers.
* Lookup priority is last override, local, last default.
* ``context`` is the destination node, ``origin`` is the source declaration
  node, ``root`` is the source tree root, and ``global_root`` is the top-level
  Resolver root.
* ``path`` is an immutable PathRef for the destination context. Nodes expose
  ``path``, ``source_path``, ``root``, and ``file`` metadata attributes.
* Path helpers: ``normalize_path``, ``absolute_path``, ``relative_path``,
  ``path_of``, ``source_path_of``, ``at``, ``get``, ``root_of``, and
  ``source_file``.
* Lists directly under ``$``, ``@``, or ``^`` resolve string items lazily in
  native-expression, text-template, or script mode respectively.
* ``import_yaml`` (alias ``import``) and ``import_json`` load lazy trees whose
  source ``root`` and file metadata remain independent.
* ``Resolver.messages`` collects ``warning`` and ``hint`` diagnostics; use
  ``emit_messages=False`` or ``treat_warnings_as_errors=True`` to control them.

Requires: jinja2 >= 3.1
Optional: PyYAML >= 6.0 for YAML support.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from typing import Any, Iterator, MutableMapping, MutableSequence

from jinja2 import ChainableUndefined, StrictUndefined, Undefined, nodes, pass_context
from jinja2.exceptions import UndefinedError
from jinja2.ext import Extension
from jinja2.nativetypes import NativeEnvironment
from jinja2.runtime import Context, missing
from jinja2.sandbox import SandboxedEnvironment

__all__ = [
    "JinestError",
    "JinestTemplateError",
    "JinestFunctionError",
    "JinestMessage",
    "JinestWarningError",
    "JinestMergeError",
    "JinestImportError",
    "JinestPathError",
    "PathRef",
    "Resolver",
    "resolve",
    "resolve_text",
    "resolve_file",
]

__version__ = "0.11.0"

_INTERNAL_SCOPE = "__jinest_scope__"
_INTERNAL_FUNCTION_LOCALS = "__jinest_function_locals__"
_RESERVED_NAMES = {
    _INTERNAL_SCOPE,
    _INTERNAL_FUNCTION_LOCALS,
    "root",
    "global_root",
    "context",
    "origin",
    "_",
    "path",
    "import_yaml",
    "import",
    "import_json",
    "normalize_path",
    "absolute_path",
    "relative_path",
    "path_of",
    "source_path_of",
    "at",
    "get",
    "root_of",
    "source_file",
}
_MERGE_RE = re.compile(
    r"^<<(?P<order>\d*)(?P<override>!)?(?P<mode>[$^])$"
)
_MISSING = object()
_EMPTY_MAPPING: Mapping[Any, Any] = {}
_NODE_META_NAMES = {"path", "source_path", "root", "file"}


class _UnicodeHexBytes(str):
    """A byte string whose JSON form must use one ``\\uHHHH`` per byte."""


def _encode_json_string(value: str) -> str:
    if isinstance(value, _UnicodeHexBytes):
        return '"' + "".join(f"\\u{ord(char):04X}" for char in value) + '"'
    return json.encoder.encode_basestring(value)


class _JinestJSONEncoder(json.JSONEncoder):
    """JSON encoder preserving the requested pure Unicode byte escapes."""

    def encode(self, value: Any) -> str:
        # JSONEncoder special-cases top-level strings before iterencode().
        if isinstance(value, _UnicodeHexBytes):
            return _encode_json_string(value)
        return super().encode(value)

    def iterencode(self, value: Any, _one_shot: bool = False) -> Iterator[str]:
        markers = {} if self.check_circular else None

        def floatstr(number: float) -> str:
            if math.isfinite(number):
                return float.__repr__(number)
            if not self.allow_nan:
                raise ValueError(
                    f"Out of range float values are not JSON compliant: {number!r}"
                )
            if math.isnan(number):
                return "NaN"
            return "Infinity" if number > 0 else "-Infinity"

        # Python 3.13 expects a prepared indent string here; older versions
        # also accept it, so normalize before using this private helper.
        indent = self.indent
        if indent is not None and not isinstance(indent, str):
            indent = " " * indent

        encoder = json.encoder._make_iterencode(
            markers,
            self.default,
            _encode_json_string,
            indent,
            floatstr,
            self.key_separator,
            self.item_separator,
            self.sort_keys,
            self.skipkeys,
            _one_shot,
        )
        return encoder(value, 0)


class JinestError(Exception):
    """Base error raised by Jinest."""


class JinestTemplateError(JinestError):
    """A Jinja expression or template could not be evaluated."""


class JinestFunctionError(JinestError):
    """A Jinest template function could not be called safely."""


@dataclass(frozen=True, slots=True)
class JinestMessage:
    """A diagnostic collected while resolving a Jinest tree.

    ``level`` is currently ``"warning"`` or ``"hint"`` and ``msg`` is the
    human-readable diagnostic text.  Message objects are intentionally small
    and immutable so callers can safely inspect or copy ``Resolver.messages``.
    """

    level: str
    msg: str


class JinestWarningError(JinestError):
    """Warnings were configured to abort resolution."""


class JinestMergeError(JinestError):
    """A merge directive did not produce a mapping."""


class JinestImportError(JinestError):
    """An imported JSON/YAML file could not be loaded."""


class JinestPathError(JinestError):
    """A Jinest path could not be parsed or resolved."""


@dataclass(frozen=True, slots=True)
class _Source:
    """Raw template container together with the resolver that owns its root."""

    resolver: "Resolver"
    raw: Any
    source_path: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class _LayerSpec:
    source_key: str
    template: Any
    order: int
    position: int
    override: bool
    mode: str


@dataclass(frozen=True, slots=True)
class _Candidate:
    source_key: Any
    template: Any
    mode: str  # concrete, native, text


@dataclass(frozen=True, slots=True)
class _FunctionParameter:
    name: str
    default: str | None


@dataclass(frozen=True, slots=True)
class _FunctionSpec:
    name: str
    source_key: str
    template: Any
    mode: str  # native, text, script
    parameters: tuple[_FunctionParameter, ...]


@dataclass(slots=True)
class _MapSchema:
    raw: Mapping[Any, Any]
    defaults: tuple[_LayerSpec, ...]
    overrides: tuple[_LayerSpec, ...]
    functions: tuple[_FunctionSpec, ...]


_FUNCTION_MODES = {"$": "native", "@": "text", "^": "script"}
_FUNCTION_MODE_MARKERS = {value: key for key, value in _FUNCTION_MODES.items()}


@dataclass(frozen=True, slots=True)
class _MappingKeyEntry:
    """One source mapping key after raw/dynamic-key normalization."""

    source_key: Any
    key: Any
    raw: bool = False
    dynamic: bool = False


def _raw_key(key: Any) -> str | None:
    """Return the literal key represented by one trailing raw-key marker."""
    if isinstance(key, str) and key.endswith("`"):
        return key[:-1]
    return None


def _inline_directive(value: Any) -> tuple[str, str] | None:
    """Parse a scalar ``=<mode>`` directive without evaluating it."""
    if (
        isinstance(value, str)
        and len(value) >= 2
        and value[0] == "="
        and value[1] in _FUNCTION_MODES
    ):
        return _FUNCTION_MODES[value[1]], value[2:]
    return None


def _escaped_inline_literal(value: Any) -> str | None:
    """Return an escaped inline directive as a literal string, if applicable."""
    if (
        isinstance(value, str)
        and len(value) >= 3
        and value[0] == "`"
        and value[1] == "="
        and value[2] in _FUNCTION_MODES
    ):
        return value[1:]
    return None


def _literal_syntax_key(key: Any) -> bool:
    """Whether a source key must bypass every Jinest key parser."""
    return (
        _raw_key(key) is not None
        or _inline_directive(key) is not None
        or _escaped_inline_literal(key) is not None
    )


def _parse_function_declaration(key: Any, template: Any = _MISSING) -> _FunctionSpec | None:
    """Parse a suffixed function declaration key without evaluating defaults."""
    if not isinstance(key, str) or _literal_syntax_key(key):
        return None
    marker = key[-1:] if key else ""
    if marker not in _FUNCTION_MODES:
        if (
            re.match(r"^[A-Za-z_]\w*\(", key)
            and not key.endswith(")")
            and not key.endswith(":")
        ):
            raise JinestError(f"Malformed function declaration {key!r}")
        return None

    declaration = key[:-1]
    opening = declaration.find("(")
    if opening <= 0 or not declaration.endswith(")"):
        if opening > 0:
            raise JinestError(f"Malformed function declaration {key!r}")
        return None

    name = declaration[:opening]
    if not name.isidentifier():
        raise JinestError(f"Malformed function declaration {key!r}: invalid name")
    parameters_text = declaration[opening + 1 : -1]
    source = f"def __jinest_function__({parameters_text}):\n    pass\n"
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise JinestError(
            f"Malformed function declaration {key!r}: {exc.msg}"
        ) from exc

    function_node = tree.body[0]
    if not isinstance(function_node, ast.FunctionDef):
        raise JinestError(f"Malformed function declaration {key!r}")
    arguments = function_node.args
    if (
        arguments.posonlyargs
        or arguments.vararg is not None
        or arguments.kwonlyargs
        or arguments.kwarg is not None
    ):
        raise JinestError(
            f"Unsupported parameters in function declaration {key!r}: "
            "*args, **kwargs, positional-only, and keyword-only parameters are unsupported"
        )
    if any(argument.annotation is not None for argument in arguments.args):
        raise JinestError(
            f"Type annotations are unsupported in function declaration {key!r}"
        )

    positional = arguments.args
    defaults_start = len(positional) - len(arguments.defaults)
    parameters: list[_FunctionParameter] = []
    seen: set[str] = set()
    for index, argument in enumerate(positional):
        if argument.arg in seen:
            raise JinestError(
                f"Duplicate parameter {argument.arg!r} in function declaration {key!r}"
            )
        seen.add(argument.arg)
        default = None
        if index >= defaults_start:
            default_node = arguments.defaults[index - defaults_start]
            default = ast.get_source_segment(source, default_node)
            if default is None:
                raise JinestError(
                    f"Could not read default for parameter {argument.arg!r} "
                    f"in function declaration {key!r}"
                )
        parameters.append(_FunctionParameter(argument.arg, default))

    return _FunctionSpec(
        name=name,
        source_key=key,
        template=template,
        mode=_FUNCTION_MODES[marker],
        parameters=tuple(parameters),
    )


class JinestFunction:
    """Safe lazy callable exposed to Jinja for one function declaration."""

    __slots__ = ("_jinest_owner", "_jinest_spec", "_jinest_source")

    def __init__(
        self,
        owner: "Resolver",
        spec: _FunctionSpec,
        source: _Source,
    ) -> None:
        object.__setattr__(self, "_jinest_owner", owner)
        object.__setattr__(self, "_jinest_spec", spec)
        object.__setattr__(self, "_jinest_source", source)

    @pass_context
    def __call__(self, jinja_context: Context, *args: Any, **kwargs: Any) -> Any:
        return object.__getattribute__(self, "_jinest_owner")._invoke_function(
            self, jinja_context, args, kwargs
        )

    def __repr__(self) -> str:
        spec = object.__getattribute__(self, "_jinest_spec")
        return f"<JinestFunction {spec.name}>"


class PathRef:
    """Immutable, Jinja-friendly reference to a value inside a Jinest tree.

    Attribute/item access extends the path. ``_`` moves to the parent and
    ``absolute`` converts a relative path to an absolute one. PathRef itself
    intentionally exposes no node metadata: ``path.file`` addresses a field
    named ``file``; use ``at(path).file`` to inspect the target node metadata.
    """

    __slots__ = (
        "_jinest_owner",
        "_jinest_root",
        "_jinest_kind",
        "_jinest_segments",
        "_jinest_relative",
        "_jinest_anchor_segments",
        "_jinest_up",
    )

    def __init__(
        self,
        owner: "Resolver",
        root: "_ContainerProxy",
        kind: str,
        segments: tuple[Any, ...] = (),
        *,
        relative: bool = False,
        anchor_segments: tuple[Any, ...] | None = None,
        up: int = 0,
    ) -> None:
        if kind not in {"global", "source"}:
            raise ValueError(f"Unsupported path root kind: {kind!r}")
        object.__setattr__(self, "_jinest_owner", owner)
        object.__setattr__(self, "_jinest_root", root)
        object.__setattr__(self, "_jinest_kind", kind)
        object.__setattr__(self, "_jinest_segments", tuple(segments))
        object.__setattr__(self, "_jinest_relative", relative)
        object.__setattr__(
            self,
            "_jinest_anchor_segments",
            tuple(anchor_segments or ()),
        )
        object.__setattr__(self, "_jinest_up", up)

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_jinest_") or name.startswith("__"):
            return object.__getattribute__(self, name)
        if name == "_":
            return object.__getattribute__(self, "_jinest_parent")()
        if name == "absolute":
            return object.__getattribute__(self, "_jinest_absolute")()
        return object.__getattribute__(self, "_jinest_append")(name)

    def __getitem__(self, key: Any) -> "PathRef":
        return self._jinest_append(key)

    def __str__(self) -> str:
        if object.__getattribute__(self, "_jinest_relative"):
            up = object.__getattribute__(self, "_jinest_up")
            segments = object.__getattribute__(self, "_jinest_segments")
            prefix = ".".join("_" for _ in range(up)) if up else "context"
            return _format_path_segments(prefix, segments)

        kind = object.__getattribute__(self, "_jinest_kind")
        prefix = "global_root" if kind == "global" else "root"
        return _format_path_segments(
            prefix,
            object.__getattribute__(self, "_jinest_segments"),
        )

    def __repr__(self) -> str:
        return f"PathRef({str(self)!r})"

    def __hash__(self) -> int:
        absolute = self._jinest_absolute()
        return hash(
            (
                id(object.__getattribute__(absolute, "_jinest_root")),
                object.__getattribute__(absolute, "_jinest_kind"),
                object.__getattribute__(absolute, "_jinest_segments"),
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PathRef):
            return False
        left = self._jinest_absolute()
        right = other._jinest_absolute()
        return (
            object.__getattribute__(left, "_jinest_root")
            is object.__getattribute__(right, "_jinest_root")
            and object.__getattribute__(left, "_jinest_kind")
            == object.__getattribute__(right, "_jinest_kind")
            and object.__getattribute__(left, "_jinest_segments")
            == object.__getattribute__(right, "_jinest_segments")
        )

    def _jinest_append(self, key: Any) -> "PathRef":
        return PathRef(
            object.__getattribute__(self, "_jinest_owner"),
            object.__getattribute__(self, "_jinest_root"),
            object.__getattribute__(self, "_jinest_kind"),
            object.__getattribute__(self, "_jinest_segments") + (key,),
            relative=object.__getattribute__(self, "_jinest_relative"),
            anchor_segments=object.__getattribute__(self, "_jinest_anchor_segments"),
            up=object.__getattribute__(self, "_jinest_up"),
        )

    def _jinest_parent(self) -> "PathRef":
        segments = object.__getattribute__(self, "_jinest_segments")
        if object.__getattribute__(self, "_jinest_relative"):
            if segments:
                return PathRef(
                    object.__getattribute__(self, "_jinest_owner"),
                    object.__getattribute__(self, "_jinest_root"),
                    object.__getattribute__(self, "_jinest_kind"),
                    segments[:-1],
                    relative=True,
                    anchor_segments=object.__getattribute__(
                        self, "_jinest_anchor_segments"
                    ),
                    up=object.__getattribute__(self, "_jinest_up"),
                )
            return PathRef(
                object.__getattribute__(self, "_jinest_owner"),
                object.__getattribute__(self, "_jinest_root"),
                object.__getattribute__(self, "_jinest_kind"),
                (),
                relative=True,
                anchor_segments=object.__getattribute__(
                    self, "_jinest_anchor_segments"
                ),
                up=object.__getattribute__(self, "_jinest_up") + 1,
            )

        if not segments:
            raise JinestPathError("Cannot move above path root")
        return PathRef(
            object.__getattribute__(self, "_jinest_owner"),
            object.__getattribute__(self, "_jinest_root"),
            object.__getattribute__(self, "_jinest_kind"),
            segments[:-1],
        )

    def _jinest_absolute(self) -> "PathRef":
        if not object.__getattribute__(self, "_jinest_relative"):
            return self
        anchor = object.__getattribute__(self, "_jinest_anchor_segments")
        up = object.__getattribute__(self, "_jinest_up")
        if up > len(anchor):
            raise JinestPathError("Relative path moves above its root")
        segments = anchor[: len(anchor) - up] + object.__getattribute__(
            self, "_jinest_segments"
        )
        return PathRef(
            object.__getattribute__(self, "_jinest_owner"),
            object.__getattribute__(self, "_jinest_root"),
            object.__getattribute__(self, "_jinest_kind"),
            segments,
        )


class _ScriptReturn(Exception):
    def __init__(self, value: Any) -> None:
        super().__init__()
        self.value = value


class _ReturnExtension(Extension):
    tags = {"return"}

    def parse(self, parser: Any) -> nodes.Node:
        token = next(parser.stream)
        if parser.stream.current.type == "block_end":
            value = nodes.Const(None)
        else:
            value = parser.parse_expression()
        return nodes.ExprStmt(self.call_method("_return", [value])).set_lineno(
            token.lineno
        )

    def _return(self, value: Any = None) -> None:
        raise _ScriptReturn(value)


class _JinestContext(Context):
    """Jinja context that lazily falls back to the current Jinest node."""

    def resolve_or_missing(self, key: str) -> Any:
        parent = self.parent
        scope = self.vars.get(_INTERNAL_SCOPE, parent.get(_INTERNAL_SCOPE))
        function_locals = self.vars.get(
            _INTERNAL_FUNCTION_LOCALS,
            parent.get(_INTERNAL_FUNCTION_LOCALS),
        )

        # Function arguments and script locals are explicit context variables.
        # They must shadow attachments and fields, while ordinary Jinest lookup
        # retains its historical field-before-global priority.
        if function_locals is not None:
            value = super().resolve_or_missing(key)
            if value is not missing:
                return value

        if key not in _RESERVED_NAMES and isinstance(scope, _ContainerProxy):
            value = scope._jinest_resolve_name(key)
            if value is not _MISSING:
                return value

        value = super().resolve_or_missing(key)
        if value is not missing:
            return value
        return missing


class _SandboxedNativeEnvironment(SandboxedEnvironment, NativeEnvironment):
    """NativeEnvironment with Jinja sandbox checks enabled."""

    def is_safe_attribute(self, obj: Any, attr: str, value: Any) -> bool:
        if attr == "_" and isinstance(obj, (_ContainerProxy, PathRef)):
            return True
        if isinstance(obj, _ContainerProxy) and attr in _NODE_META_NAMES:
            return True
        if isinstance(obj, PathRef) and attr == "absolute":
            return True
        return super().is_safe_attribute(obj, attr, value)


class _ContainerProxy:
    """A raw container bound to a destination parent/path."""

    __slots__ = (
        "_jinest_owner",
        "_jinest_source",
        "_jinest_parent",
        "_jinest_path",
        "_jinest_path_kind",
        "_jinest_children",
    )

    def __init__(
        self,
        owner: "Resolver",
        source: _Source,
        parent: "_ContainerProxy | None",
        path: tuple[Any, ...],
        path_kind: str = "global",
    ) -> None:
        object.__setattr__(self, "_jinest_owner", owner)
        object.__setattr__(self, "_jinest_source", source)
        object.__setattr__(self, "_jinest_parent", parent)
        object.__setattr__(self, "_jinest_path", path)
        object.__setattr__(self, "_jinest_path_kind", path_kind)
        object.__setattr__(self, "_jinest_children", {})

    def __getattribute__(self, name: str) -> Any:
        if name == "_":
            return object.__getattribute__(self, "_jinest_parent")
        if name == "path":
            owner = object.__getattribute__(self, "_jinest_owner")
            source = object.__getattribute__(self, "_jinest_source")
            kind = object.__getattribute__(self, "_jinest_path_kind")
            root = (
                owner._global_owner.root
                if kind == "global"
                else source.resolver._source_root
            )
            return PathRef(
                owner,
                root,
                kind,
                object.__getattribute__(self, "_jinest_path"),
            )
        if name == "source_path":
            source = object.__getattribute__(self, "_jinest_source")
            return PathRef(
                source.resolver,
                source.resolver._source_root,
                "source",
                source.source_path,
            )
        if name == "root":
            source = object.__getattribute__(self, "_jinest_source")
            return source.resolver._source_root
        if name == "file":
            source = object.__getattribute__(self, "_jinest_source")
            path = source.resolver.source_path
            return str(path) if path is not None else None
        return object.__getattribute__(self, name)

    def __repr__(self) -> str:
        path = object.__getattribute__(self, "_jinest_path")
        kind = object.__getattribute__(self, "_jinest_path_kind")
        prefix = "global_root" if kind == "global" else "root"
        return f"<{type(self).__name__} path={_format_path_segments(prefix, path)}>"

    def __str__(self) -> str:
        owner = object.__getattribute__(self, "_jinest_owner")
        return str(owner._to_plain(self, active=set()))

    def _jinest_resolve_name(self, key: str) -> Any:
        return _MISSING


class _MappingProxy(_ContainerProxy, Mapping):
    """Lazy mapping with default, local, and override layers."""

    __slots__ = (
        "_jinest_resolved",
        "_jinest_public_resolved",
        "_jinest_layer_cache",
        "_jinest_key_indexes",
    )

    def __init__(
        self,
        owner: "Resolver",
        source: _Source,
        parent: _ContainerProxy | None,
        path: tuple[Any, ...],
        path_kind: str = "global",
    ) -> None:
        super().__init__(owner, source, parent, path, path_kind)
        object.__setattr__(self, "_jinest_resolved", {})
        object.__setattr__(self, "_jinest_public_resolved", {})
        object.__setattr__(self, "_jinest_layer_cache", {})
        object.__setattr__(self, "_jinest_key_indexes", {})

    def __getattribute__(self, name: str) -> Any:
        if name == "_" or name in _NODE_META_NAMES:
            return super().__getattribute__(name)

        if not name.startswith("_jinest_") and not name.startswith("__"):
            owner = object.__getattribute__(self, "_jinest_owner")
            if owner._scope_has_logical(self, name):
                return owner._get_field(self, name)

        return object.__getattribute__(self, name)

    def _jinest_resolve_name(self, key: str) -> Any:
        owner = object.__getattribute__(self, "_jinest_owner")
        if owner._scope_has_logical(self, key):
            return owner._get_field(self, key)
        return _MISSING

    def __getitem__(self, key: Any) -> Any:
        owner = object.__getattribute__(self, "_jinest_owner")
        if isinstance(key, PathRef):
            return owner._at_path(key, anchor=self)
        return owner._get_field(self, key)

    def __iter__(self) -> Iterator[Any]:
        owner = object.__getattribute__(self, "_jinest_owner")
        yield from owner._public_keys(self)

    def __len__(self) -> int:
        owner = object.__getattribute__(self, "_jinest_owner")
        return len(owner._public_keys(self))


class _SequenceProxy(_ContainerProxy, Sequence):
    """Lazy sequence, optionally resolving each string item as Jinja."""

    __slots__ = (
        "_jinest_item_mode",
        "_jinest_key_context",
        "_jinest_resolved",
    )

    def __init__(
        self,
        owner: "Resolver",
        source: _Source,
        parent: _ContainerProxy | None,
        path: tuple[Any, ...],
        *,
        item_mode: str | None = None,
        key_context: tuple[Any, Any, str | None] | None = None,
        path_kind: str = "global",
    ) -> None:
        super().__init__(owner, source, parent, path, path_kind)
        if item_mode not in {None, "native", "text", "script"}:
            raise ValueError(f"Unsupported sequence item mode: {item_mode!r}")
        object.__setattr__(self, "_jinest_item_mode", item_mode)
        object.__setattr__(self, "_jinest_key_context", key_context)
        object.__setattr__(self, "_jinest_resolved", {})

    def _jinest_resolve_name(self, key: str) -> Any:
        # Unqualified variables in array items come from the nearest mapping.
        parent = object.__getattribute__(self, "_jinest_parent")
        while isinstance(parent, _SequenceProxy):
            parent = object.__getattribute__(parent, "_jinest_parent")
        if isinstance(parent, _MappingProxy):
            return parent._jinest_resolve_name(key)
        return _MISSING

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, PathRef):
            owner = object.__getattribute__(self, "_jinest_owner")
            return owner._at_path(index, anchor=self)
        source = object.__getattribute__(self, "_jinest_source")
        raw = source.raw
        owner = object.__getattribute__(self, "_jinest_owner")

        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(raw)))]

        normalized = index if index >= 0 else len(raw) + index
        if normalized < 0 or normalized >= len(raw):
            raise IndexError(index)

        resolved = object.__getattribute__(self, "_jinest_resolved")
        if normalized in resolved:
            return resolved[normalized]

        # Same cycle rule as mapping fields: recursive access sees None.
        resolved[normalized] = None
        object.__getattribute__(self, "_jinest_children").pop(normalized, None)

        try:
            item = raw[normalized]
            mode = object.__getattribute__(self, "_jinest_item_mode")
            item_path = object.__getattribute__(self, "_jinest_path") + (normalized,)

            key_context = object.__getattribute__(self, "_jinest_key_context")
            explicit_directive = _inline_directive(item)
            explicit, explicit_value = owner._resolve_inline_scalar(
                self,
                item,
                origin_source=source,
                source_key=normalized,
                context_path=item_path,
                keyname=None if key_context is None else key_context[0],
                effective_key=None if key_context is None else key_context[1],
            )
            if explicit:
                if mode is not None and explicit_directive is not None:
                    owner._record_message(
                        "warning",
                        f"Inline directive at "
                        f"{_format_path_segments('global_root', item_path)} takes "
                        f"precedence over legacy {_FUNCTION_MODE_MARKERS[mode]} array mode",
                        dedupe_key=("inline-array", id(raw), normalized),
                    )
                value = owner._bind_child(
                    self,
                    normalized,
                    explicit_value,
                    origin=source.resolver,
                    source_path=source.source_path + (normalized,),
                )
            elif mode is not None and isinstance(item, str):
                value = owner._render(
                    self,
                    item,
                    mode=mode,
                    origin_source=source,
                    source_path=source.source_path + (normalized,),
                    context_path=item_path,
                    keyname=None if key_context is None else key_context[0],
                    effective_key=None if key_context is None else key_context[1],
                    keymode=None if key_context is None else key_context[2],
                )
                value = owner._bind_child(
                    self,
                    normalized,
                    value,
                    origin=source.resolver,
                    source_path=source.source_path + (normalized,),
                )
            else:
                value = owner._bind_child(
                    self,
                    normalized,
                    item,
                    origin=source.resolver,
                    source_path=source.source_path + (normalized,),
                )
        except Exception:
            resolved.pop(normalized, None)
            object.__getattribute__(self, "_jinest_children").pop(normalized, None)
            raise

        resolved[normalized] = value
        return value

    def __len__(self) -> int:
        source = object.__getattribute__(self, "_jinest_source")
        return len(source.raw)


class Resolver:
    """Resolve a structured Python tree containing lazy Jinja fields."""

    def __init__(
        self,
        data: Any,
        *,
        in_place: bool = False,
        strict: bool = True,
        sandboxed: bool = True,
        globals: Mapping[str, Any] | None = None,
        filters: Mapping[str, Any] | None = None,
        source_path: str | os.PathLike[str] | None = None,
        base_dir: str | os.PathLike[str] | None = None,
        import_roots: Sequence[str | os.PathLike[str]] | None = None,
        function_max_depth: int = 100,
        emit_messages: bool = True,
        treat_warnings_as_errors: bool = False,
        _import_chain: tuple[Path, ...] | None = None,
        _global_owner: "Resolver | None" = None,
    ) -> None:
        self.in_place = in_place
        self.strict = strict
        self.sandboxed = sandboxed
        if not isinstance(function_max_depth, int) or function_max_depth < 1:
            raise ValueError("function_max_depth must be a positive integer")
        self.function_max_depth = function_max_depth
        if not isinstance(emit_messages, bool):
            raise TypeError("emit_messages must be a boolean")
        if not isinstance(treat_warnings_as_errors, bool):
            raise TypeError("treat_warnings_as_errors must be a boolean")
        self.emit_messages = emit_messages
        self.treat_warnings_as_errors = treat_warnings_as_errors
        self._global_owner = _global_owner or self
        if self._global_owner is self:
            self.messages: list[JinestMessage] = []
            self._message_keys: set[tuple[str, str]] = set()
            self._emitted_message_count = 0
        else:
            self.messages = self._global_owner.messages
            self._message_keys = self._global_owner._message_keys
        self._function_depth = 0
        self._function_stack: list[str] = []
        self._user_globals = dict(globals or {})
        self._user_filters = dict(filters or {})
        self._original = data
        self.data = data if in_place else copy.deepcopy(data)
        self._schema_cache: dict[int, _MapSchema] = {}
        self._import_cache: dict[tuple[Path, str], Resolver] = {}
        self._source_view_cache: dict[
            tuple[int, tuple[Any, ...]], _ContainerProxy
        ] = {}
        self.source_path = Path(source_path).expanduser().resolve() if source_path else None
        if base_dir is not None:
            self.base_dir = Path(base_dir).expanduser().resolve()
        elif self.source_path is not None:
            self.base_dir = self.source_path.parent
        else:
            self.base_dir = Path.cwd().resolve()

        if import_roots is None:
            self.import_roots: tuple[Path, ...] | None = None
        else:
            if isinstance(import_roots, (str, os.PathLike)):
                raise TypeError("import_roots must be a sequence of directories")
            roots = tuple(Path(root).expanduser().resolve() for root in import_roots)
            for root in roots:
                if not root.is_dir():
                    raise ValueError(f"Import root is not a directory: {root}")
            self.import_roots = roots

        if _import_chain is not None:
            self._import_chain = _import_chain
        elif self.source_path is not None:
            self._import_chain = (self.source_path,)
        else:
            self._import_chain = ()

        environment_type = _SandboxedNativeEnvironment if sandboxed else NativeEnvironment
        undefined_type = StrictUndefined if strict else ChainableUndefined
        self.environment = environment_type(undefined=undefined_type)
        self.script_environment = environment_type(
            undefined=undefined_type,
            extensions=[_ReturnExtension],
            line_statement_prefix="%",
        )
        self.environment.context_class = _JinestContext
        self.script_environment.context_class = _JinestContext

        import_yaml_fn = lambda path: self._import_tree(path, "yaml")
        import_json_fn = lambda path: self._import_tree(path, "json")
        @pass_context
        def normalize_path_fn(jinja_context: Context, value: Any) -> PathRef:
            return self._normalize_path(value, frame=jinja_context)

        @pass_context
        def absolute_path_fn(
            jinja_context: Context,
            value: Any,
            anchor: Any = _MISSING,
            **kwargs: Any,
        ) -> PathRef:
            if "root" in kwargs and anchor is _MISSING:
                anchor = kwargs.pop("root")
            if kwargs:
                raise TypeError(f"Unexpected arguments: {', '.join(kwargs)}")
            return self._absolute_path(value, anchor=anchor, frame=jinja_context)

        @pass_context
        def relative_path_fn(
            jinja_context: Context,
            target: Any,
            base: Any = _MISSING,
            **kwargs: Any,
        ) -> PathRef:
            if "path" in kwargs and base is _MISSING:
                base = kwargs.pop("path")
            if kwargs:
                raise TypeError(f"Unexpected arguments: {', '.join(kwargs)}")
            return self._relative_path(target, base=base, frame=jinja_context)

        @pass_context
        def path_of_fn(jinja_context: Context, node: Any) -> PathRef:
            return self._path_of(node, source=False)

        @pass_context
        def source_path_of_fn(jinja_context: Context, node: Any) -> PathRef:
            return self._path_of(node, source=True)

        @pass_context
        def at_fn(
            jinja_context: Context,
            target: Any,
            anchor: Any = _MISSING,
        ) -> Any:
            return self._at(target, anchor=anchor, frame=jinja_context)

        @pass_context
        def get_fn(
            jinja_context: Context,
            target: Any,
            default: Any = None,
            anchor: Any = _MISSING,
        ) -> Any:
            try:
                return self._at(target, anchor=anchor, frame=jinja_context)
            except (KeyError, IndexError, TypeError, JinestPathError, UndefinedError):
                return default

        @pass_context
        def root_of_fn(jinja_context: Context, node: Any) -> Any:
            return self._root_of(node)

        @pass_context
        def source_file_fn(jinja_context: Context, node: Any) -> Any:
            return self._source_file(node)

        builtins = {
            "import_yaml": import_yaml_fn,
            "import": import_yaml_fn,
            "import_json": import_json_fn,
            "normalize_path": normalize_path_fn,
            "absolute_path": absolute_path_fn,
            "relative_path": relative_path_fn,
            "path_of": path_of_fn,
            "source_path_of": source_path_of_fn,
            "at": at_fn,
            "get": get_fn,
            "root_of": root_of_fn,
            "source_file": source_file_fn,
        }
        self.environment.globals.update(builtins)
        self.environment.filters.update(builtins)
        self.environment.globals.update(self._user_globals)
        self.environment.filters.update(self._user_filters)
        self.script_environment.globals.update(builtins)
        self.script_environment.filters.update(builtins)
        self.script_environment.globals.update(self._user_globals)
        self.script_environment.filters.update(self._user_filters)

        self._reset_root_views()

    def _record_message(
        self,
        level: str,
        msg: str,
        *,
        dedupe_key: tuple[Any, ...] | None = None,
    ) -> None:
        """Add one deduplicated diagnostic to the shared resolver message list."""
        if level not in {"warning", "hint"}:
            raise ValueError(f"Unsupported message level: {level!r}")
        key = dedupe_key or (level, msg)
        if key in self._message_keys:
            return
        self._message_keys.add(key)
        self.messages.append(JinestMessage(level, msg))

    def _flush_messages(self) -> None:
        """Print newly collected diagnostics, if stderr output is enabled."""
        owner = self._global_owner
        if not owner.emit_messages:
            owner._emitted_message_count = len(owner.messages)
            return
        pending = owner.messages[owner._emitted_message_count:]
        for message in pending:
            print(f"jinest: {message.level}: {message.msg}", file=sys.stderr)
        owner._emitted_message_count = len(owner.messages)

    def _finalize_messages(self) -> None:
        """Apply warning policy and emit diagnostics after successful resolution."""
        owner = self._global_owner
        warnings = [message for message in owner.messages if message.level == "warning"]
        if owner.treat_warnings_as_errors and warnings:
            details = "\n".join(message.msg for message in warnings)
            raise JinestWarningError(
                f"Warnings treated as errors ({len(warnings)}):\n{details}"
            )
        self._flush_messages()

    def _reset_root_views(self) -> None:
        """Rebuild destination/source roots after initialization or in-place resolve."""
        root_kind = "global" if self._global_owner is self else "source"
        self.root = self._wrap(
            self.data,
            parent=None,
            path=(),
            origin=self,
            source_path=(),
            path_kind=root_kind,
        )
        if isinstance(self.root, _ContainerProxy):
            if root_kind == "source":
                self._source_root = self.root
            else:
                self._source_root = self._wrap(
                    self.data,
                    parent=None,
                    path=(),
                    origin=self,
                    source_path=(),
                    path_kind="source",
                )
            root_source = object.__getattribute__(self._source_root, "_jinest_source")
            self._source_view_cache[(id(root_source.raw), ())] = self._source_root
        else:
            # Scalars have no template scope or source-view metadata.
            self._source_root = self.root
        self.global_root = self._global_owner.root
        if isinstance(self.data, Mapping):
            # Make root diagnostics available immediately through ``messages``;
            # nested mappings are discovered when their lazy scopes are visited.
            self._schema(self.data)

    def resolve(self) -> Any:
        """Fully materialize the lazy tree into ordinary Python values."""
        try:
            result = self._to_plain(self.root, active=set())
        except Exception:
            # Diagnostics discovered before a rendering error are still useful
            # to API callers and CLI users.
            self._flush_messages()
            raise
        self._finalize_messages()

        if self.in_place:
            if isinstance(self._original, MutableMapping) and isinstance(result, Mapping):
                self._original.clear()
                self._original.update(result)
                self.data = self._original
            elif (
                isinstance(self._original, MutableSequence)
                and not isinstance(self._original, (str, bytes, bytearray))
                and isinstance(result, list)
            ):
                self._original[:] = result
                self.data = self._original
            else:
                self.data = result

            self._schema_cache.clear()
            self._source_view_cache.clear()
            self._reset_root_views()
            return self.data

        return result

    # ------------------------------------------------------------------
    # Binding and source ownership
    # ------------------------------------------------------------------

    def _wrap(
        self,
        value: Any,
        parent: _ContainerProxy | None,
        path: tuple[Any, ...],
        *,
        origin: "Resolver | None" = None,
        source_path: tuple[Any, ...] | None = None,
        sequence_item_mode: str | None = None,
        sequence_key_context: tuple[Any, Any, str | None] | None = None,
        path_kind: str = "global",
    ) -> Any:
        if isinstance(value, _ContainerProxy):
            source = object.__getattribute__(value, "_jinest_source")
        elif isinstance(value, Mapping):
            source = _Source(origin or self, value, tuple(source_path or ()))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            source = _Source(origin or self, value, tuple(source_path or ()))
        else:
            return self._copy_scalar(value)

        if isinstance(source.raw, Mapping):
            return _MappingProxy(self, source, parent, path, path_kind)
        if isinstance(source.raw, Sequence) and not isinstance(
            source.raw, (str, bytes, bytearray)
        ):
            if isinstance(value, _SequenceProxy):
                if sequence_item_mode is None:
                    sequence_item_mode = object.__getattribute__(value, "_jinest_item_mode")
                if sequence_key_context is None:
                    sequence_key_context = object.__getattribute__(
                        value, "_jinest_key_context"
                    )
            return _SequenceProxy(
                self,
                source,
                parent,
                path,
                item_mode=sequence_item_mode,
                key_context=sequence_key_context,
                path_kind=path_kind,
            )
        return copy.deepcopy(source.raw)

    def _bind_child(
        self,
        parent: _ContainerProxy,
        key: Any,
        value: Any,
        *,
        origin: "Resolver | None" = None,
        source_path: tuple[Any, ...] | None = None,
        sequence_item_mode: str | None = None,
        sequence_key_context: tuple[Any, Any, str | None] | None = None,
    ) -> Any:
        if not self._is_container(value):
            return self._copy_scalar(value)

        if isinstance(value, _ContainerProxy):
            source = object.__getattribute__(value, "_jinest_source")
            raw = source.raw
            source_origin = source.resolver
        else:
            raw = value
            source_origin = origin or self
            source_origin_path = tuple(source_path or ())

        if isinstance(value, _ContainerProxy):
            source_origin_path = source.source_path

        children = object.__getattribute__(parent, "_jinest_children")
        cache_token = (
            id(raw),
            id(source_origin),
            source_origin_path,
            sequence_item_mode,
            sequence_key_context,
        )
        cached = children.get(key)
        if cached is not None and cached[0] == cache_token:
            return cached[1]

        path = object.__getattribute__(parent, "_jinest_path") + (key,)
        path_kind = object.__getattribute__(parent, "_jinest_path_kind")
        proxy = self._wrap(
            value,
            parent=parent,
            path=path,
            origin=source_origin,
            source_path=source_origin_path,
            sequence_item_mode=sequence_item_mode,
            sequence_key_context=sequence_key_context,
            path_kind=path_kind,
        )
        children[key] = (cache_token, proxy)
        return proxy

    @staticmethod
    def _is_container(value: Any) -> bool:
        return isinstance(value, _ContainerProxy) or isinstance(value, Mapping) or (
            isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
        )

    def _copy_scalar(self, value: Any) -> Any:
        """Copy supported scalar values and apply strictness to invalid ones."""
        if isinstance(value, PathRef):
            return value
        supported = value is None or isinstance(
            value,
            (str, bool, int, bytes, bytearray, date, time),
        )
        if isinstance(value, float):
            supported = math.isfinite(value)
        if supported:
            return copy.deepcopy(value)
        if not self.strict:
            return None
        raise JinestError(
            f"Unsupported scalar value of type {type(value).__name__}"
        )

    # ------------------------------------------------------------------
    # Mapping schema and lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_key(key: Any) -> re.Match[str] | None:
        if not isinstance(key, str) or _literal_syntax_key(key):
            return None
        return _MERGE_RE.fullmatch(key)

    @classmethod
    def _template_key(cls, key: Any) -> tuple[Any, str] | None:
        if (
            not isinstance(key, str)
            or _literal_syntax_key(key)
            or cls._merge_key(key)
        ):
            return None
        if key.endswith("^"):
            return key[:-1], "script"
        if key.endswith("$"):
            return key[:-1], "native"
        if key.endswith("@"):
            return key[:-1], "text"
        return None

    def _mapping_entries(
        self,
        source: _Source,
        bind: _MappingProxy,
    ) -> tuple[_MappingKeyEntry, ...]:
        """Build a destination-bound index of static, raw, and dynamic keys."""
        indexes = object.__getattribute__(bind, "_jinest_key_indexes")
        cache_key = (id(source.resolver), id(source.raw), source.source_path)
        cached = indexes.get(cache_key, _MISSING)
        if cached is not _MISSING:
            # While dynamic keys are being evaluated, static keys are already
            # present. This lets ``=$base`` use a sibling ``base`` field.
            return tuple(cached) if isinstance(cached, list) else cached

        entries: list[_MappingKeyEntry] = []
        seen: dict[Any, _MappingKeyEntry] = {}
        dynamic_source_keys: list[Any] = []
        source_positions = {
            key: position for position, key in enumerate(source.raw)
        }

        def is_literal_field(entry: _MappingKeyEntry) -> bool:
            if entry.raw or entry.dynamic or not isinstance(entry.source_key, str):
                return True
            return (
                self._merge_key(entry.source_key) is None
                and _parse_function_declaration(entry.source_key) is None
                and self._template_key(entry.source_key) is None
            )

        def add(entry: _MappingKeyEntry) -> None:
            # ``value$`` is a declaration for logical ``value``. It may coexist
            # with raw/dynamic literal key ``value$`` and is therefore excluded
            # from final-key collision detection.
            if is_literal_field(entry):
                previous = seen.get(entry.key)
                if previous is not None:
                    kind = (
                        "dynamic mapping key"
                        if entry.dynamic or previous.dynamic
                        else "mapping key"
                    )
                    raise JinestError(
                        f"Duplicate {kind} {entry.key!r} from "
                        f"{previous.source_key!r} and {entry.source_key!r}"
                    )
                seen[entry.key] = entry
            entries.append(entry)

        try:
            # Index all non-dynamic keys before rendering expressions. Their
            # availability preserves normal lazy sibling lookup semantics.
            for source_key in source.raw:
                if _inline_directive(source_key) is not None:
                    dynamic_source_keys.append(source_key)
                    continue
                raw_key = _raw_key(source_key)
                escaped = _escaped_inline_literal(source_key)
                if raw_key is not None:
                    add(_MappingKeyEntry(source_key, raw_key, raw=True))
                elif escaped is not None:
                    add(_MappingKeyEntry(source_key, escaped, raw=True))
                else:
                    add(_MappingKeyEntry(source_key, source_key))

            # Expose the static prefix during dynamic-key rendering.
            indexes[cache_key] = entries
            for source_key in dynamic_source_keys:
                mode, template = _inline_directive(source_key)  # known above
                marker = _FUNCTION_MODE_MARKERS[mode]
                key = self._render(
                    bind,
                    template,
                    mode=mode,
                    origin_source=source,
                    source_key=source_key,
                    keyname=None,
                    effective_key=source_key,
                    keymode=marker,
                )
                if not isinstance(key, str):
                    raise JinestError(
                        f"Dynamic mapping key {source_key!r} resolved to "
                        f"{type(key).__name__}, expected a string"
                    )
                add(_MappingKeyEntry(source_key, key, dynamic=True))
        except Exception:
            indexes.pop(cache_key, None)
            raise

        entries.sort(key=lambda entry: source_positions[entry.source_key])
        result = tuple(entries)
        indexes[cache_key] = result
        return result

    def _resolve_inline_scalar(
        self,
        scope: _ContainerProxy,
        value: Any,
        *,
        origin_source: _Source,
        source_key: Any,
        context_path: tuple[Any, ...] | None = None,
        keyname: Any | None = None,
        effective_key: Any | None = None,
    ) -> tuple[bool, Any]:
        """Resolve one explicit scalar directive or its leading-backtick escape."""
        escaped = _escaped_inline_literal(value)
        if escaped is not None:
            return True, escaped
        directive = _inline_directive(value)
        if directive is None:
            return False, value
        mode, template = directive
        return True, self._render(
            scope,
            template,
            mode=mode,
            origin_source=origin_source,
            source_key=source_key,
            context_path=context_path,
            keyname=keyname,
            effective_key=effective_key,
            keymode=_FUNCTION_MODE_MARKERS[mode],
        )

    def _record_schema_messages(self, raw: Mapping[Any, Any]) -> None:
        """Report declarations that are present but cannot become effective."""
        declarations: dict[tuple[str, bool], list[tuple[int, str]]] = {}
        priority = {"": 0, "^": 1, "$": 2, "@": 3}
        for key in raw:
            if (
                not isinstance(key, str)
                or _literal_syntax_key(key)
                or self._merge_key(key)
            ):
                continue
            if _parse_function_declaration(key) is not None:
                continue
            mode = ""
            base = key
            if key.endswith(("^", "$", "@")):
                mode = key[-1]
                base = key[:-1]
            hidden = base.startswith(".")
            logical = base[1:] if hidden else base
            if not isinstance(logical, str):
                continue
            declarations.setdefault((logical, hidden), []).append(
                (priority[mode], key)
            )

        for (logical, hidden), variants in declarations.items():
            if len(variants) < 2:
                continue
            variants.sort(key=lambda item: item[0])
            winner = variants[0][1]
            for _, suppressed in variants[1:]:
                self._record_message(
                    "warning",
                    f"Field {winner!r} suppresses {suppressed!r}; local priority is "
                    "name > name^ > name$ > name@",
                    dedupe_key=("warning", id(raw), winner, suppressed),
                )

        public_names = {logical for (logical, hidden) in declarations if not hidden}
        hidden_names = {logical for (logical, hidden) in declarations if hidden}
        for logical in sorted(public_names & hidden_names):
            self._record_message(
                "hint",
                f"Hidden field '.{logical}' takes priority over field {logical!r} "
                "in template calculations; the public field remains in the final dump",
                dedupe_key=("hint", id(raw), logical),
            )

    def _schema(self, raw: Mapping[Any, Any]) -> _MapSchema:
        raw_id = id(raw)
        cached = self._schema_cache.get(raw_id)
        if cached is not None and cached.raw is raw:
            return cached

        defaults: list[_LayerSpec] = []
        overrides: list[_LayerSpec] = []
        functions: list[_FunctionSpec] = []
        function_names: set[str] = set()
        for position, (key, template) in enumerate(raw.items()):
            function = _parse_function_declaration(key, template)
            if function is not None:
                if function.name in function_names:
                    raise JinestError(
                        f"Duplicate function declaration {function.name!r} "
                        f"at {_format_path_segments('root', (function.source_key,))}"
                    )
                function_names.add(function.name)
                functions.append(function)
                continue

            match = self._merge_key(key)
            if match is None:
                continue
            order_text = match.group("order")
            order = int(order_text) if order_text else 0
            spec = _LayerSpec(
                source_key=key,
                template=template,
                order=order,
                position=position,
                override=bool(match.group("override")),
                mode="script" if match.group("mode") == "^" else "native",
            )
            (overrides if spec.override else defaults).append(spec)

        # A function and an ordinary declaration cannot share one logical name.
        ordinary_names: dict[str, Any] = {}
        for key in raw:
            if (
                _literal_syntax_key(key)
                or _parse_function_declaration(key) is not None
            ):
                continue
            if self._merge_key(key):
                continue
            template_info = self._template_key(key)
            logical = template_info[0] if template_info else key
            if isinstance(logical, str) and logical.startswith("."):
                logical = logical[1:]
            if isinstance(logical, str):
                ordinary_names.setdefault(logical, key)
        for function in functions:
            conflict = ordinary_names.get(function.name)
            if conflict is not None:
                raise JinestError(
                    f"Function {function.name!r} at {function.source_key!r} "
                    f"conflicts with field declaration {conflict!r}"
                )

        defaults.sort(key=lambda item: (item.order, item.position))
        overrides.sort(key=lambda item: (item.order, item.position))
        schema = _MapSchema(
            raw, tuple(defaults), tuple(overrides), tuple(functions)
        )
        self._schema_cache[raw_id] = schema
        self._record_schema_messages(raw)
        return schema

    def _local_candidate(
        self,
        source: _Source,
        bind: _MappingProxy,
        key: Any,
        *,
        hidden: bool,
    ) -> _Candidate | None:
        """Return one local declaration from the hidden or public namespace."""
        source_key = f".{key}" if hidden and isinstance(key, str) else key
        entries = self._mapping_entries(source, bind)
        for concrete in entries:
            if concrete.key != source_key:
                continue
            if (
                concrete.raw
                or concrete.dynamic
                or self._template_key(concrete.source_key) is None
            ):
                if (
                    self._merge_key(concrete.source_key) is None
                    and _parse_function_declaration(concrete.source_key) is None
                ):
                    return _Candidate(
                        concrete.source_key,
                        source.raw[concrete.source_key],
                        "concrete",
                    )

        if isinstance(source_key, str):
            for suffix, mode in (("^", "script"), ("$", "native"), ("@", "text")):
                physical = f"{source_key}{suffix}"
                for entry in entries:
                    if entry.key == physical and not entry.raw and not entry.dynamic:
                        return _Candidate(
                            entry.source_key,
                            source.raw[entry.source_key],
                            mode,
                        )
        return None

    @staticmethod
    def _hidden_name(key: Any) -> bool:
        return isinstance(key, str) and not key.startswith(".")

    def _local_function(self, raw: Mapping[Any, Any], key: Any) -> _FunctionSpec | None:
        if not isinstance(key, str) or key.startswith("."):
            return None
        schema = self._schema(raw)
        for function in schema.functions:
            if function.name == key:
                return function
        return None

    def _function_value(self, source: _Source, spec: _FunctionSpec) -> JinestFunction:
        return JinestFunction(self, spec, source)

    def _scope_has_logical(self, scope: _MappingProxy, key: Any) -> bool:
        resolved = object.__getattribute__(scope, "_jinest_resolved")
        if key in resolved:
            return True
        source = object.__getattribute__(scope, "_jinest_source")
        return self._contains(source, key, bind=scope, active=set(), hidden=None)

    def _contains(
        self,
        source: _Source,
        key: Any,
        *,
        bind: _MappingProxy,
        active: set[tuple[Any, ...]],
        hidden: bool | None,
    ) -> bool:
        if hidden is None:
            if self._hidden_name(key) and self._contains(
                source, key, bind=bind, active=active, hidden=True
            ):
                return True
            return self._contains(source, key, bind=bind, active=active, hidden=False)

        token = (
            id(source.resolver),
            id(source.raw),
            self._hashable_key(key),
            hidden,
        )
        if token in active:
            return False
        active.add(token)
        try:
            schema = self._schema(source.raw)

            for layer in reversed(schema.overrides):
                layer_source = self._evaluate_layer(bind, source, layer)
                if self._contains(
                    layer_source, key, bind=bind, active=active, hidden=hidden
                ):
                    return True

            if not hidden and self._local_function(source.raw, key) is not None:
                return True
            if self._local_candidate(source, bind, key, hidden=hidden) is not None:
                return True

            for layer in reversed(schema.defaults):
                layer_source = self._evaluate_layer(bind, source, layer)
                if self._contains(
                    layer_source, key, bind=bind, active=active, hidden=hidden
                ):
                    return True
            return False
        finally:
            active.remove(token)

    def _get_field(self, scope: _MappingProxy, key: Any) -> Any:
        return self._get_cached_field(scope, key, public=False)

    def _get_public_field(self, scope: _MappingProxy, key: Any) -> Any:
        """Resolve a key for serialization, ignoring its hidden declaration."""
        return self._get_cached_field(scope, key, public=True)

    def _get_cached_field(
        self, scope: _MappingProxy, key: Any, *, public: bool
    ) -> Any:
        cache_name = "_jinest_public_resolved" if public else "_jinest_resolved"
        resolved = object.__getattribute__(scope, cache_name)
        if key in resolved:
            return resolved[key]

        resolved[key] = None
        object.__getattribute__(scope, "_jinest_children").pop(key, None)
        try:
            source = object.__getattribute__(scope, "_jinest_source")
            value = self._lookup(
                source,
                key,
                bind=scope,
                active=set(),
                hidden=False if public else None,
            )
            if value is _MISSING:
                raise KeyError(key)
        except Exception:
            resolved.pop(key, None)
            object.__getattribute__(scope, "_jinest_children").pop(key, None)
            raise

        resolved[key] = value
        return value

    def _lookup(
        self,
        source: _Source,
        key: Any,
        *,
        bind: _MappingProxy,
        active: set[tuple[Any, ...]],
        hidden: bool | None,
    ) -> Any:
        """Look up a key, preferring the hidden namespace when requested."""
        if hidden is None:
            if self._hidden_name(key):
                value = self._lookup(
                    source, key, bind=bind, active=active, hidden=True
                )
                if value is not _MISSING:
                    return value
            return self._lookup(source, key, bind=bind, active=active, hidden=False)

        token = (
            id(source.resolver),
            id(source.raw),
            self._hashable_key(key),
            hidden,
        )
        if token in active:
            return _MISSING
        active.add(token)
        try:
            schema = self._schema(source.raw)

            # Reverse lookup of: defaults -> local -> overrides.
            for layer in reversed(schema.overrides):
                layer_source = self._evaluate_layer(bind, source, layer)
                value = self._lookup(
                    layer_source, key, bind=bind, active=active, hidden=hidden
                )
                if value is not _MISSING:
                    return value

            if not hidden:
                function = self._local_function(source.raw, key)
                if function is not None:
                    return self._function_value(source, function)

            candidate = self._local_candidate(source, bind, key, hidden=hidden)
            if candidate is not None:
                return self._resolve_candidate(candidate, source, bind, key)

            for layer in reversed(schema.defaults):
                layer_source = self._evaluate_layer(bind, source, layer)
                value = self._lookup(
                    layer_source, key, bind=bind, active=active, hidden=hidden
                )
                if value is not _MISSING:
                    return value

            return _MISSING
        finally:
            active.remove(token)

    def _resolve_candidate(
        self,
        candidate: _Candidate,
        source: _Source,
        bind: _MappingProxy,
        logical_key: Any,
    ) -> Any:
        candidate_source_path = source.source_path + (candidate.source_key,)
        if candidate.mode == "concrete":
            explicit, value = self._resolve_inline_scalar(
                bind,
                candidate.template,
                origin_source=source,
                source_key=candidate.source_key,
                keyname=logical_key,
                effective_key=candidate.source_key,
            )
            return self._bind_child(
                bind,
                logical_key,
                value if explicit else candidate.template,
                origin=source.resolver,
                source_path=candidate_source_path,
            )

        explicit_directive = _inline_directive(candidate.template)
        explicit, explicit_value = self._resolve_inline_scalar(
            bind,
            candidate.template,
            origin_source=source,
            source_key=candidate.source_key,
            keyname=logical_key,
            effective_key=candidate.source_key,
        )
        if explicit:
            if explicit_directive is not None:
                path = object.__getattribute__(bind, "_jinest_path") + (logical_key,)
                self._record_message(
                    "warning",
                    f"Inline directive at "
                    f"{_format_path_segments('global_root', path)} takes precedence "
                    f"over {_FUNCTION_MODE_MARKERS[candidate.mode]} field mode",
                    dedupe_key=(
                        "inline-field",
                        id(source.raw),
                        candidate.source_key,
                    ),
                )
            return self._bind_child(
                bind,
                logical_key,
                explicit_value,
                origin=source.resolver,
                source_path=candidate_source_path,
            )

        if isinstance(candidate.template, Sequence) and not isinstance(
            candidate.template, (str, bytes, bytearray)
        ):
            return self._bind_child(
                bind,
                logical_key,
                candidate.template,
                origin=source.resolver,
                source_path=candidate_source_path,
                sequence_item_mode=candidate.mode,
                sequence_key_context=(
                    logical_key,
                    candidate.source_key,
                    {"native": "$", "text": "@", "script": "^"}[candidate.mode],
                ),
            )

        value = self._render(
            bind,
            candidate.template,
            mode=candidate.mode,
            origin_source=source,
            source_key=candidate.source_key,
            keyname=logical_key,
            effective_key=candidate.source_key,
            keymode={"native": "$", "text": "@", "script": "^"}[candidate.mode],
        )
        if candidate.mode in {"native", "script"}:
            return self._bind_child(
                bind,
                logical_key,
                value,
                origin=source.resolver,
                source_path=candidate_source_path,
            )
        return value

    def _evaluate_layer(
        self,
        bind: _MappingProxy,
        owner_source: _Source,
        layer: _LayerSpec,
    ) -> _Source:
        cache = object.__getattribute__(bind, "_jinest_layer_cache")
        cache_key = (id(owner_source.resolver), id(owner_source.raw), layer.source_key)
        if cache_key in cache:
            cached = cache[cache_key]
            # A recursively requested layer is temporarily empty, mirroring
            # field cycles resolving to None.
            if cached is None:
                return _Source(
                    owner_source.resolver,
                    _EMPTY_MAPPING,
                    owner_source.source_path + (layer.source_key,),
                )
            return cached

        cache[cache_key] = None
        try:
            value = self._render(
                bind,
                layer.template,
                mode=layer.mode,
                origin_source=owner_source,
                source_key=layer.source_key,
            )

            if value is None:
                result = _Source(
                    owner_source.resolver,
                    _EMPTY_MAPPING,
                    owner_source.source_path + (layer.source_key,),
                )
            elif isinstance(value, _MappingProxy):
                result = object.__getattribute__(value, "_jinest_source")
            elif isinstance(value, Mapping):
                result = _Source(
                    owner_source.resolver,
                    value,
                    owner_source.source_path + (layer.source_key,),
                )
            else:
                path = object.__getattribute__(bind, "_jinest_path") + (layer.source_key,)
                raise JinestMergeError(
                    f"Merge {_format_path_segments('global_root', path)} produced "
                    f"{type(value).__name__}, expected a mapping"
                )
        except Exception:
            cache.pop(cache_key, None)
            raise

        cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # Source views, node metadata, and path operations
    # ------------------------------------------------------------------

    def _raw_source_at(self, path: tuple[Any, ...]) -> Any:
        value = self.data
        for part in path:
            if isinstance(value, Mapping):
                value = value[part]
            elif isinstance(value, Sequence) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                value = value[part]
            else:
                raise JinestPathError(
                    f"Source path {_format_path_segments('root', path)} "
                    "passes through a scalar"
                )
        return value

    def _source_view(self, source: _Source) -> _ContainerProxy:
        cache_key = (id(source.raw), source.source_path)
        cached = source.resolver._source_view_cache.get(cache_key)
        if cached is not None:
            return cached

        parent: _ContainerProxy | None = None
        if source.source_path:
            parent_path = source.source_path[:-1]
            try:
                parent_raw = source.resolver._raw_source_at(parent_path)
            except (KeyError, IndexError, TypeError, JinestPathError):
                parent_raw = None
            if source.resolver._is_container(parent_raw):
                parent = source.resolver._source_view(
                    _Source(source.resolver, parent_raw, parent_path)
                )

        proxy = source.resolver._wrap(
            source.raw,
            parent=parent,
            path=source.source_path,
            origin=source.resolver,
            source_path=source.source_path,
            path_kind="source",
        )
        if not isinstance(proxy, _ContainerProxy):
            raise JinestPathError("Origin context must be a mapping or sequence")
        source.resolver._source_view_cache[cache_key] = proxy
        return proxy

    @staticmethod
    def _frame_get(frame: Context | None, name: str) -> Any:
        if frame is None:
            return _MISSING
        value = frame.resolve_or_missing(name)
        return _MISSING if value is missing else value

    def _default_anchor(self, frame: Context | None) -> _ContainerProxy:
        value = self._frame_get(frame, "context")
        if isinstance(value, _ContainerProxy):
            return value
        value = self._frame_get(frame, _INTERNAL_SCOPE)
        if isinstance(value, _ContainerProxy):
            return value
        raise JinestPathError("A relative path requires a Jinest context anchor")

    def _path_from_node(self, node: _ContainerProxy, *, source: bool) -> PathRef:
        return node.source_path if source else node.path

    def _path_of(self, node: Any, *, source: bool) -> PathRef:
        if not isinstance(node, _ContainerProxy):
            raise JinestPathError(
                f"path_of() expects a Jinest mapping/list node, got "
                f"{type(node).__name__}"
            )
        return self._path_from_node(node, source=source)

    def _root_of(self, node: Any) -> Any:
        if not isinstance(node, _ContainerProxy):
            raise JinestPathError(
                f"root_of() expects a Jinest mapping/list node, got "
                f"{type(node).__name__}"
            )
        return node.root

    def _source_file(self, node: Any) -> str | None:
        if not isinstance(node, _ContainerProxy):
            raise JinestPathError(
                f"source_file() expects a Jinest mapping/list node, got "
                f"{type(node).__name__}"
            )
        return node.file

    def _anchor_path(
        self,
        anchor: Any,
        *,
        frame: Context | None,
    ) -> PathRef:
        if anchor is _MISSING:
            anchor = self._default_anchor(frame)
        if isinstance(anchor, _ContainerProxy):
            return anchor.path
        if isinstance(anchor, PathRef):
            return anchor._jinest_absolute()
        if isinstance(anchor, str):
            return self._parse_path(anchor, frame=frame, anchor=_MISSING)._jinest_absolute()
        raise JinestPathError(
            f"Path anchor must be a Jinest node or PathRef, got "
            f"{type(anchor).__name__}"
        )

    def _parse_path(
        self,
        text: str,
        *,
        frame: Context | None,
        anchor: Any = _MISSING,
    ) -> PathRef:
        try:
            expression = ast.parse(text.strip(), mode="eval").body
        except (SyntaxError, ValueError) as exc:
            raise JinestPathError(f"Invalid path {text!r}: {exc}") from exc

        def named_root(name: str) -> PathRef | None:
            if name == "path":
                value = self._frame_get(frame, "path")
                if not isinstance(value, PathRef):
                    raise JinestPathError("path is unavailable outside evaluation")
                return value
            if name == "global_root":
                value = self._frame_get(frame, "global_root")
                if value is _MISSING:
                    value = self._global_owner.root
                if not isinstance(value, _ContainerProxy):
                    raise JinestPathError("global_root is not a Jinest node")
                return value.path
            if name == "root":
                value = self._frame_get(frame, "root")
                if not isinstance(value, _ContainerProxy):
                    raise JinestPathError("root is unavailable outside evaluation")
                return value.path
            if name in {"context", "origin"}:
                value = self._frame_get(frame, name)
                if not isinstance(value, _ContainerProxy):
                    raise JinestPathError(f"{name} is unavailable outside evaluation")
                return value.path
            return None

        def build(node: ast.AST) -> PathRef:
            if isinstance(node, ast.Name):
                root_path = named_root(node.id)
                if root_path is not None:
                    return root_path

                base = self._anchor_path(anchor, frame=frame)
                if node.id == "_":
                    return PathRef(
                        object.__getattribute__(base, "_jinest_owner"),
                        object.__getattribute__(base, "_jinest_root"),
                        object.__getattribute__(base, "_jinest_kind"),
                        (),
                        relative=True,
                        anchor_segments=object.__getattribute__(
                            base, "_jinest_segments"
                        ),
                        up=1,
                    )
                return PathRef(
                    object.__getattribute__(base, "_jinest_owner"),
                    object.__getattribute__(base, "_jinest_root"),
                    object.__getattribute__(base, "_jinest_kind"),
                    (node.id,),
                    relative=True,
                    anchor_segments=object.__getattribute__(
                        base, "_jinest_segments"
                    ),
                )

            if isinstance(node, ast.Attribute):
                base = build(node.value)
                if node.attr == "_":
                    return base._jinest_parent()
                return base[node.attr]

            if isinstance(node, ast.Subscript):
                base = build(node.value)
                slice_node = node.slice
                if isinstance(slice_node, ast.Constant):
                    key = slice_node.value
                elif isinstance(slice_node, ast.UnaryOp) and isinstance(
                    slice_node.op, ast.USub
                ) and isinstance(slice_node.operand, ast.Constant) and isinstance(
                    slice_node.operand.value, (int, float)
                ):
                    key = -slice_node.operand.value
                else:
                    raise JinestPathError(
                        "Path indexes must be literal strings, integers, or numbers"
                    )
                return base[key]

            raise JinestPathError(
                "Paths may contain only names, attributes, and literal indexes"
            )

        return build(expression)

    def _normalize_path(
        self,
        value: Any,
        *,
        frame: Context | None,
        anchor: Any = _MISSING,
    ) -> PathRef:
        if isinstance(value, PathRef):
            return value
        if isinstance(value, _ContainerProxy):
            return value.path
        if isinstance(value, str):
            return self._parse_path(value, frame=frame, anchor=anchor)
        raise JinestPathError(
            f"Expected a path string, PathRef, or Jinest node; got "
            f"{type(value).__name__}"
        )

    def _absolute_path(
        self,
        value: Any,
        *,
        anchor: Any = _MISSING,
        frame: Context | None,
    ) -> PathRef:
        path = self._normalize_path(value, frame=frame, anchor=anchor)
        if object.__getattribute__(path, "_jinest_relative") and anchor is not _MISSING:
            path = self._reanchor_relative_path(
                path,
                self._anchor_path(anchor, frame=frame),
            )
        return path._jinest_absolute()

    @staticmethod
    def _reanchor_relative_path(path: PathRef, anchor_path: PathRef) -> PathRef:
        if (
            object.__getattribute__(anchor_path, "_jinest_root")
            is not object.__getattribute__(path, "_jinest_root")
            or object.__getattribute__(anchor_path, "_jinest_kind")
            != object.__getattribute__(path, "_jinest_kind")
        ):
            raise JinestPathError(
                "Relative path and anchor belong to different root spaces"
            )
        return PathRef(
            object.__getattribute__(anchor_path, "_jinest_owner"),
            object.__getattribute__(anchor_path, "_jinest_root"),
            object.__getattribute__(anchor_path, "_jinest_kind"),
            object.__getattribute__(path, "_jinest_segments"),
            relative=True,
            anchor_segments=object.__getattribute__(
                anchor_path, "_jinest_segments"
            ),
            up=object.__getattribute__(path, "_jinest_up"),
        )

    def _relative_path(
        self,
        target: Any,
        *,
        base: Any = _MISSING,
        frame: Context | None,
    ) -> PathRef:
        target_path = self._absolute_path(target, frame=frame)
        base_path = self._anchor_path(base, frame=frame)

        target_root = object.__getattribute__(target_path, "_jinest_root")
        base_root = object.__getattribute__(base_path, "_jinest_root")
        target_kind = object.__getattribute__(target_path, "_jinest_kind")
        base_kind = object.__getattribute__(base_path, "_jinest_kind")
        if target_root is not base_root or target_kind != base_kind:
            raise JinestPathError(
                "Cannot build a relative path between different root spaces"
            )

        target_segments = object.__getattribute__(target_path, "_jinest_segments")
        base_segments = object.__getattribute__(base_path, "_jinest_segments")
        common = 0
        for left, right in zip(target_segments, base_segments):
            if left != right:
                break
            common += 1

        return PathRef(
            object.__getattribute__(target_path, "_jinest_owner"),
            target_root,
            target_kind,
            target_segments[common:],
            relative=True,
            anchor_segments=base_segments,
            up=len(base_segments) - common,
        )

    def _at(
        self,
        target: Any,
        *,
        anchor: Any = _MISSING,
        frame: Context | None,
    ) -> Any:
        path = self._normalize_path(target, frame=frame, anchor=anchor)
        if object.__getattribute__(path, "_jinest_relative") and anchor is not _MISSING:
            path = self._reanchor_relative_path(
                path,
                self._anchor_path(anchor, frame=frame),
            )
        return self._at_path(path)

    def _at_path(self, path: PathRef, *, anchor: Any = None) -> Any:
        if not isinstance(path, PathRef):
            raise TypeError("_at_path() expects PathRef")
        if object.__getattribute__(path, "_jinest_relative") and anchor is not None:
            if not isinstance(anchor, _ContainerProxy):
                raise JinestPathError("Relative node[path] access requires node anchor")
            absolute = self._reanchor_relative_path(path, anchor.path)._jinest_absolute()
        else:
            absolute = path._jinest_absolute()
        current: Any = object.__getattribute__(absolute, "_jinest_root")
        segments = object.__getattribute__(absolute, "_jinest_segments")

        for part in segments:
            if isinstance(current, _MappingProxy):
                current = current[part]
            elif isinstance(current, _SequenceProxy):
                if not isinstance(part, int):
                    raise JinestPathError(
                        f"Sequence index must be integer, got {part!r}"
                    )
                current = current[part]
            elif isinstance(current, Mapping):
                current = current[part]
            elif isinstance(current, Sequence) and not isinstance(
                current, (str, bytes, bytearray)
            ):
                current = current[part]
            else:
                raise JinestPathError(
                    f"Path {absolute} passes through scalar "
                    f"{type(current).__name__}"
                )
        return current

    # ------------------------------------------------------------------
    # Jinja rendering and imports
    # ------------------------------------------------------------------

    def _invoke_function(
        self,
        function: JinestFunction,
        jinja_context: Context,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        spec = object.__getattribute__(function, "_jinest_spec")
        source = object.__getattribute__(function, "_jinest_source")
        call_scope = jinja_context.vars.get(_INTERNAL_SCOPE)
        if not isinstance(call_scope, _ContainerProxy):
            parent = jinja_context.parent
            call_scope = parent.get(_INTERNAL_SCOPE)
        if not isinstance(call_scope, _ContainerProxy):
            call_scope = self.root if isinstance(self.root, _ContainerProxy) else None
        if not isinstance(call_scope, _ContainerProxy):
            raise JinestFunctionError(
                f"Function {spec.name!r} has no Jinest call-site context"
            )

        call_path = object.__getattribute__(call_scope, "_jinest_path")
        declaration_path = source.source_path + (spec.source_key,)
        display_declaration = _format_path_segments("root", declaration_path)
        display_call = _format_path_segments("global_root", call_path)
        call_chain = " -> ".join(self._function_stack + [spec.name])
        if self._function_depth >= self.function_max_depth:
            raise JinestFunctionError(
                f"Jinest function recursion limit exceeded ({self.function_max_depth}) "
                f"at {display_call}: {call_chain}"
            )

        parameters = spec.parameters
        parameter_names = {parameter.name for parameter in parameters}
        if len(args) > len(parameters):
            raise JinestFunctionError(
                f"Function {spec.name!r} at {display_declaration} received "
                f"too many positional arguments at {display_call}"
            )
        unknown = [name for name in kwargs if name not in parameter_names]
        if unknown:
            raise JinestFunctionError(
                f"Function {spec.name!r} at {display_declaration} received "
                f"unknown argument {unknown[0]!r} at {display_call}"
            )

        bound: dict[str, Any] = {}
        for parameter, value in zip(parameters, args):
            bound[parameter.name] = value
        for name, value in kwargs.items():
            if name in bound:
                raise JinestFunctionError(
                    f"Function {spec.name!r} at {display_declaration} received "
                    f"duplicate argument {name!r} at {display_call}"
                )
            bound[name] = value

        marker = _FUNCTION_MODE_MARKERS[spec.mode]
        metadata = {
            "keyname": spec.name,
            "effective_key": spec.source_key,
            "keymode": marker,
        }

        self._function_depth += 1
        self._function_stack.append(spec.name)
        try:
            for parameter in parameters:
                if parameter.name in bound:
                    continue
                if parameter.default is None:
                    raise JinestFunctionError(
                        f"Function {spec.name!r} at {display_declaration} is missing "
                        f"required argument {parameter.name!r} at {display_call}"
                    )
                try:
                    bound[parameter.name] = self._render(
                        call_scope,
                        parameter.default,
                        mode="native",
                        origin_source=source,
                        source_path=declaration_path + (
                            f"default:{parameter.name}",
                        ),
                        context_path=call_path,
                        keyname=metadata["keyname"],
                        effective_key=metadata["effective_key"],
                        keymode=metadata["keymode"],
                        local_vars=bound,
                    )
                except JinestError as exc:
                    raise JinestFunctionError(
                        f"Failed to evaluate default for function {spec.name!r} "
                        f"at {display_declaration}, call site {display_call}: {exc}"
                    ) from exc

            try:
                return self._render(
                    call_scope,
                    spec.template,
                    mode=spec.mode,
                    origin_source=source,
                    source_path=declaration_path,
                    context_path=call_path,
                    keyname=metadata["keyname"],
                    effective_key=metadata["effective_key"],
                    keymode=metadata["keymode"],
                    local_vars=bound,
                )
            except JinestFunctionError:
                raise
            except JinestError as exc:
                raise JinestFunctionError(
                    f"Function {spec.name!r} failed at {display_call}; "
                    f"declaration {display_declaration}; call chain {call_chain}: {exc}"
                ) from exc
            except Exception as exc:
                raise JinestFunctionError(
                    f"Function {spec.name!r} failed at {display_call}; "
                    f"declaration {display_declaration}; call chain {call_chain}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        finally:
            self._function_stack.pop()
            self._function_depth -= 1

    @staticmethod
    def _normalize_multiline_returns(template: str) -> str:
        """Join line-statement ``return`` expressions spanning bracketed lines."""
        lines = template.splitlines(keepends=True)
        index = 0
        while index < len(lines):
            line = lines[index]
            content = line.rstrip("\r\n")
            marker = content.find("%")
            if marker < 0:
                index += 1
                continue
            statement = content[marker + 1 :].lstrip()
            if not statement.startswith("return"):
                index += 1
                continue
            expression = statement[len("return") :].lstrip()
            if not expression or _balanced_delimiters(expression) <= 0:
                index += 1
                continue

            combined = content[: marker + 1] + " return " + expression
            consumed = index + 1
            balance = _balanced_delimiters(expression)
            while consumed < len(lines) and balance > 0:
                continuation = lines[consumed].rstrip("\r\n")
                continuation_marker = continuation.find("%")
                if continuation_marker < 0:
                    break
                fragment = continuation[continuation_marker + 1 :].strip()
                combined += " " + fragment
                balance = _balanced_delimiters(combined.split(" return ", 1)[1])
                consumed += 1
            if balance == 0:
                newline = "\n" if line.endswith("\n") else ""
                lines[index] = combined + newline
                for blank in range(index + 1, consumed):
                    lines[blank] = newline
                index = consumed
                continue
            index += 1
        return "".join(lines)

    def _render(
        self,
        scope: _ContainerProxy,
        template: Any,
        *,
        mode: str,
        origin_source: _Source,
        source_key: Any | None = None,
        source_path: tuple[Any, ...] | None = None,
        context_path: tuple[Any, ...] | None = None,
        keyname: Any | None = None,
        effective_key: Any | None = None,
        keymode: str | None = None,
        local_vars: Mapping[str, Any] | None = None,
    ) -> Any:
        if mode not in {"text", "native", "script"}:
            raise ValueError(f"Unsupported render mode: {mode!r}")

        scope_path = object.__getattribute__(scope, "_jinest_path")
        if source_path is None:
            source_path = (
                origin_source.source_path
                if source_key is None
                else origin_source.source_path + (source_key,)
            )
        if context_path is None:
            context_path = scope_path

        if not isinstance(template, str):
            return str(template) if mode == "text" else self._prepare_native(template)

        path_kind = object.__getattribute__(scope, "_jinest_path_kind")
        path_root = (
            self._global_owner.root
            if path_kind == "global"
            else origin_source.resolver._source_root
        )
        path_owner = self if path_kind == "global" else origin_source.resolver
        context_path_ref = PathRef(
            path_owner,
            path_root,
            path_kind,
            context_path,
        )
        origin_context = origin_source.resolver._source_view(origin_source)
        context = {
            _INTERNAL_SCOPE: scope,
            "context": scope,
            "origin": origin_context,
            "root": origin_source.resolver._source_root,
            "global_root": self._global_owner.root,
            "_": object.__getattribute__(scope, "_jinest_parent"),
            "path": context_path_ref,
            "keyname": keyname,
            "effective_key": effective_key,
            "keymode": keymode,
        }
        if local_vars is not None:
            context[_INTERNAL_FUNCTION_LOCALS] = local_vars
            context.update(local_vars)
        environment = (
            origin_source.resolver.script_environment
            if mode == "script"
            else origin_source.resolver.environment
        )
        display_path = _format_path_segments("root", source_path)

        try:
            if mode == "native":
                if "{{" in template or "{%" in template or "{#" in template:
                    raise JinestTemplateError(
                        f"Native expression {display_path} must not use "
                        "Jinja template delimiters"
                    )
                expression = environment.compile_expression(
                    template,
                    undefined_to_none=False,
                )
                result = expression(**context)
                return self._prepare_native(result)

            if mode == "script":
                compiled = environment.from_string(
                    self._normalize_multiline_returns(template)
                )
                try:
                    compiled.render(**context)
                except _ScriptReturn as returned:
                    return self._prepare_native(returned.value)
                return None

            compiled = environment.from_string(template)
            # NativeEnvironment.render() applies literal_eval to a complete
            # textual result. Generate chunks directly so @ always remains text
            # (for example, a template producing ``"hello"`` keeps its quotes).
            return "".join(str(chunk) for chunk in compiled.generate(**context))
        except _ScriptReturn as returned:
            # Defensive fallback in case an environment layer lets the return
            # escape outside the inner render call.
            return self._prepare_native(returned.value)
        except JinestError:
            raise
        except UndefinedError as exc:
            if not self.strict:
                return "" if mode == "text" else None
            raise JinestTemplateError(
                f"Failed to render {display_path}: {exc}"
            ) from exc
        except Exception as exc:
            raise JinestTemplateError(
                f"Failed to render {display_path}: {exc}"
            ) from exc

    def _prepare_native(self, value: Any) -> Any:
        if isinstance(value, Undefined):
            if not self.strict:
                return None
            value._fail_with_undefined_error()
        if isinstance(value, (_ContainerProxy, PathRef)):
            # Preserve origin/root metadata. Binding creates a fresh destination
            # proxy and therefore a fresh resolved cache.
            return value
        return self._clone_unresolved(value, active=set())

    def _clone_unresolved(self, value: Any, *, active: set[int]) -> Any:
        if isinstance(value, (_ContainerProxy, PathRef)):
            return value
        if isinstance(value, Mapping):
            value_id = id(value)
            if value_id in active:
                raise JinestError("Cyclic mapping in native expression result")
            active.add(value_id)
            try:
                return {
                    self._clone_unresolved(k, active=active):
                    self._clone_unresolved(v, active=active)
                    for k, v in value.items()
                }
            finally:
                active.remove(value_id)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            value_id = id(value)
            if value_id in active:
                raise JinestError("Cyclic sequence in native expression result")
            active.add(value_id)
            try:
                return [self._clone_unresolved(item, active=active) for item in value]
            finally:
                active.remove(value_id)
        return self._copy_scalar(value)

    def _import_tree(self, path_value: Any, format: str) -> Any:
        if isinstance(path_value, Undefined):
            path_value._fail_with_undefined_error()
        try:
            requested = Path(os.fspath(path_value)).expanduser()
        except TypeError as exc:
            raise JinestImportError(
                f"Import path must be a string or path-like value, got "
                f"{type(path_value).__name__}"
            ) from exc

        path = requested if requested.is_absolute() else self.base_dir / requested
        path = path.resolve()

        if self.import_roots is not None and not any(
            path.is_relative_to(root) for root in self.import_roots
        ):
            raise JinestImportError(
                f"Import path is outside permitted roots: {requested}"
            )

        # Import cycles follow ordinary field-cycle semantics: the currently
        # importing path resolves to None.
        if path in self._import_chain:
            return None

        cache_key = (path, format)
        cached = self._import_cache.get(cache_key)
        if cached is not None:
            return cached.root

        try:
            text = path.read_text(encoding="utf-8")
            if format == "json":
                data = json.loads(text)
            elif format == "yaml":
                data = _import_yaml_module().safe_load(text)
            else:  # internal invariant
                raise ValueError(format)
        except JinestError:
            raise
        except FileNotFoundError as exc:
            raise JinestImportError(f"Import file not found: {requested}") from exc
        except Exception as exc:
            raise JinestImportError(f"Failed to import {requested}: {exc}") from exc

        child = Resolver(
            data,
            strict=self.strict,
            sandboxed=self.sandboxed,
            globals=self._user_globals,
            filters=self._user_filters,
            source_path=path,
            import_roots=self.import_roots,
            function_max_depth=self.function_max_depth,
            emit_messages=False,
            treat_warnings_as_errors=False,
            _import_chain=self._import_chain + (path,),
            _global_owner=self._global_owner,
        )
        self._import_cache[cache_key] = child
        return child.root

    # ------------------------------------------------------------------
    # Key enumeration and materialization
    # ------------------------------------------------------------------

    def _public_keys(self, scope: _MappingProxy) -> list[Any]:
        source = object.__getattribute__(scope, "_jinest_source")
        result: list[Any] = []
        seen: set[Any] = set()
        self._collect_keys(source, bind=scope, result=result, seen=seen, active=set())
        return result

    def _collect_keys(
        self,
        source: _Source,
        *,
        bind: _MappingProxy,
        result: list[Any],
        seen: set[Any],
        active: set[tuple[int, int]],
    ) -> None:
        token = (id(source.resolver), id(source.raw))
        if token in active:
            return
        active.add(token)
        try:
            schema = self._schema(source.raw)

            for layer in schema.defaults:
                self._collect_keys(
                    self._evaluate_layer(bind, source, layer),
                    bind=bind,
                    result=result,
                    seen=seen,
                    active=active,
                )

            for entry in self._mapping_entries(source, bind):
                source_key = entry.key
                if not entry.raw and not entry.dynamic:
                    if self._merge_key(source_key):
                        continue
                    if _parse_function_declaration(source_key) is not None:
                        continue
                    template_info = self._template_key(source_key)
                    logical = template_info[0] if template_info else source_key
                else:
                    # Raw and dynamic keys produce a literal final key; their
                    # result is never fed back into Jinest's key grammar.
                    logical = source_key
                if isinstance(logical, str) and logical.startswith("."):
                    continue
                if logical not in seen:
                    seen.add(logical)
                    result.append(logical)

            for layer in schema.overrides:
                self._collect_keys(
                    self._evaluate_layer(bind, source, layer),
                    bind=bind,
                    result=result,
                    seen=seen,
                    active=active,
                )
        finally:
            active.remove(token)

    def _to_plain(self, value: Any, *, active: set[tuple[int, int]]) -> Any:
        if isinstance(value, PathRef):
            return str(value)

        if isinstance(value, _MappingProxy):
            source = object.__getattribute__(value, "_jinest_source")
            token = (id(source.resolver), id(source.raw))
            if token in active:
                raise JinestError(
                    f"Cyclic container reference at "
                    f"{_format_path(object.__getattribute__(value, '_jinest_path'))}"
                )
            active.add(token)
            try:
                return {
                    key: self._to_plain(
                        self._get_public_field(value, key), active=active
                    )
                    for key in self._public_keys(value)
                }
            finally:
                active.remove(token)

        if isinstance(value, _SequenceProxy):
            source = object.__getattribute__(value, "_jinest_source")
            token = (id(source.resolver), id(source.raw))
            if token in active:
                raise JinestError(
                    f"Cyclic container reference at "
                    f"{_format_path(object.__getattribute__(value, '_jinest_path'))}"
                )
            active.add(token)
            try:
                return [self._to_plain(value[i], active=active) for i in range(len(value))]
            finally:
                active.remove(token)

        if isinstance(value, Mapping):
            token = (0, id(value))
            if token in active:
                raise JinestError("Cyclic mapping in materialized result")
            active.add(token)
            try:
                return {
                    self._to_plain(k, active=active): self._to_plain(v, active=active)
                    for k, v in value.items()
                }
            finally:
                active.remove(token)

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            token = (0, id(value))
            if token in active:
                raise JinestError("Cyclic sequence in materialized result")
            active.add(token)
            try:
                return [self._to_plain(item, active=active) for item in value]
            finally:
                active.remove(token)

        return self._copy_scalar(value)

    @staticmethod
    def _hashable_key(key: Any) -> Any:
        try:
            hash(key)
            return key
        except TypeError:
            return (type(key).__name__, repr(key))


# ----------------------------------------------------------------------
# Public helpers
# ----------------------------------------------------------------------


def resolve(data: Any, **options: Any) -> Any:
    """Convenience wrapper: ``Resolver(data, **options).resolve()``."""
    return Resolver(data, **options).resolve()


def resolve_text(
    text: str,
    *,
    format: str = "json",
    output_format: str | None = None,
    **resolver_options: Any,
) -> str:
    """Parse JSON/YAML text, resolve it, and serialize the result."""
    source_format = format.lower()
    target_format = (output_format or source_format).lower()

    if source_format == "json":
        data = json.loads(text)
    elif source_format in {"yaml", "yml"}:
        data = _import_yaml_module().safe_load(text)
    else:
        raise ValueError(f"Unsupported input format: {format!r}")

    result = Resolver(data, **resolver_options).resolve()
    return _serialize(result, target_format)


def resolve_file(
    path: str | Path,
    *,
    output: str | Path | None = None,
    output_format: str | None = None,
    **resolver_options: Any,
) -> str:
    """Resolve a .json/.yaml/.yml file and optionally write the result."""
    input_path = Path(path).expanduser().resolve()
    source_format = _format_from_path(input_path)
    data = _parse_text(input_path.read_text(encoding="utf-8"), source_format)

    # The source path is essential for relative imports.
    resolver_options.setdefault("source_path", input_path)
    result = Resolver(data, **resolver_options).resolve()
    target_format = (output_format or source_format).lower()
    rendered = _serialize(result, target_format)

    if output is not None:
        Path(output).write_text(
            rendered + ("" if rendered.endswith("\n") else "\n"),
            encoding="utf-8",
        )
    return rendered


def _parse_text(text: str, format: str) -> Any:
    if format == "json":
        return json.loads(text)
    if format in {"yaml", "yml"}:
        return _import_yaml_module().safe_load(text)
    raise ValueError(f"Unsupported input format: {format!r}")


def _serialize(value: Any, format: str) -> str:
    if format == "json":
        return json.dumps(
            _normalize_json_value(value, active=set()),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            cls=_JinestJSONEncoder,
        )
    if format in {"yaml", "yml"}:
        return _import_yaml_module().safe_dump(
            value,
            allow_unicode=True,
            sort_keys=False,
        )
    raise ValueError(f"Unsupported output format: {format!r}")


def _normalize_json_value(value: Any, *, active: set[int]) -> Any:
    """Convert extended scalar values to standards-compliant JSON."""
    if isinstance(value, (bytes, bytearray)):
        return _UnicodeHexBytes("".join(chr(byte) for byte in value))
    if isinstance(value, (date, time)):
        return value.isoformat()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        raise JinestError(f"Non-finite float is not valid JSON: {value!r}")

    if isinstance(value, Mapping):
        value_id = id(value)
        if value_id in active:
            raise JinestError("Cyclic mapping cannot be serialized as JSON")
        active.add(value_id)
        try:
            return {
                _normalize_json_value(key, active=active):
                _normalize_json_value(item, active=active)
                for key, item in value.items()
            }
        finally:
            active.remove(value_id)

    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        value_id = id(value)
        if value_id in active:
            raise JinestError("Cyclic sequence cannot be serialized as JSON")
        active.add(value_id)
        try:
            return [
                _normalize_json_value(item, active=active)
                for item in value
            ]
        finally:
            active.remove(value_id)

    raise JinestError(
        f"Unsupported value of type {type(value).__name__} for JSON serialization"
    )


def _balanced_delimiters(value: str) -> int:
    """Return unmatched opening bracket count, ignoring quoted strings."""
    pairs = {"{": "}", "[": "]", "(": ")"}
    closing = set(pairs.values())
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    for char in value:
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in pairs:
            stack.append(pairs[char])
        elif char in closing and stack and char == stack[-1]:
            stack.pop()
    return len(stack)


def _format_from_path(path: Path) -> str:
    extension = path.suffix.lower()
    if extension == ".json":
        return "json"
    if extension in {".yaml", ".yml"}:
        return "yaml"
    raise ValueError(f"Cannot infer format from {path.name!r}")


def _import_yaml_module() -> Any:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise JinestError("YAML support requires PyYAML: pip install pyyaml") from exc
    return yaml


def _format_path_segments(prefix: str, path: tuple[Any, ...]) -> str:
    result = prefix
    for part in path:
        if isinstance(part, str) and part.isidentifier():
            result += f".{part}"
        else:
            result += f"[{part!r}]"
    return result


def _format_path(path: tuple[Any, ...]) -> str:
    """Backward-compatible internal formatter for source-root paths."""
    return _format_path_segments("root", path)


# ----------------------------------------------------------------------
# Built-in regression tests
# ----------------------------------------------------------------------


def _self_test() -> None:
    data = {
        "defaults1": {"rank": "d1", "d1": True},
        "defaults2": {"rank": "d2", "d2": True},
        "overrides1": {"rank": "o1", "o1": True},
        "overrides2": {"rank": "o2", "o2": True},
        "example": {
            "<<2!$": "root.overrides2",
            "<<2$": "root.defaults2",
            "x": 1,
            "y@": "{{ x }}",
            "z$": "x + (y | int)",
            "rank": "local",
            "<<1!$": "root.overrides1",
            "<<1$": "root.defaults1",
        },
        "priority": {
            "value@": "{{ missing.value }}",
            "value$": "40 + 2",
        },
        "cycle": {"a$": "b", "b$": "a"},
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
            "instance": {"<<$": "root.class.prototype", "var": 1},
            "other$": "root.class.prototype",
        },
        "A": {"x": 1},
        "B": {"<<$": "root.A"},
        "C": {"<<$": "root.B"},
        "var1": 7,
        "var2": 9,
        "native_array$": ["var1", "root.var2", "1", "true", 5, None, "path"],
        "text_array@": ["{{ var1 }}", "v={{ root.var2 }}", 1, True, "{{ path }}"],
        "ready_array$": "['var1', 'root.var2']",
        "path_list": [
            {"obj": {"where$": "path"}},
            {"obj": {"where@": "{{ path }}"}},
        ],
    }

    resolver = Resolver(data, emit_messages=False)
    assert resolver.root.example.rank == "o2"
    assert resolver.root.example.z == 2
    assert resolver.root.priority.value == 42
    assert resolver.root.C.x == 1
    assert resolver.root["class"].prototype.A == 1
    assert resolver.root.inherited.instance.A == 2
    assert resolver.root.inherited.instance.B == 1
    assert resolver.root.inherited.instance.C == 10
    assert resolver.root.inherited.instance.where == "global_root.inherited.instance"
    assert resolver.root.inherited.other.where == "global_root.inherited.other"
    assert resolver.root.native_array[0] == 7
    assert resolver.root.native_array[1] == 9
    assert str(resolver.root.native_array[6]) == "global_root.native_array[6]"
    assert resolver.root.text_array[4] == "global_root.text_array[4]"
    assert resolver.root.ready_array[:] == ["var1", "root.var2"]
    assert str(resolver.root.path_list[0].obj.where) == "global_root.path_list[0].obj"
    assert str(resolver.root.inherited.instance.path) == "global_root.inherited.instance"
    assert str(resolver.root.inherited.instance.source_path) == "root.inherited.instance"

    script = Resolver({
        "x": 4,
        "value^": "% set y = x * 2\n% return {'y': y}\n",
        "target": {"<<^": "% return {'a': 1}\n", "b": 2},
    }).resolve()
    assert script["value"] == {"y": 8}
    assert script["target"] == {"a": 1, "b": 2}

    result = resolver.resolve()
    assert result["cycle"] == {"a": None, "b": None}
    assert result["example"]["rank"] == "o2"
    assert result["example"]["d1"] is True
    assert result["example"]["d2"] is True
    assert result["example"]["o1"] is True
    assert result["example"]["o2"] is True

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
json_obj$: import_json('data.json')
json_value$: import_json('data.json').double
via_filter$: '"base.yaml" | import | attr("constant")'
""".lstrip(),
            encoding="utf-8",
        )

        imported = _parse_text(
            resolve_file(folder / "main.yaml", output_format="json"),
            "json",
        )
        assert imported["instance"]["absolute"] == 10
        assert imported["instance"]["relative"] == 5
        assert imported["instance"]["where"] == "global_root.instance"
        assert imported["json_obj"]["double"] == 42
        assert imported["json_value"] == 42
        assert imported["via_filter"] == 10

        # Lazy import cycle: the path already present in the import ancestry is None.
        (folder / "a.yaml").write_text("other$: import('b.yaml')\n", encoding="utf-8")
        (folder / "b.yaml").write_text("back$: import('a.yaml')\n", encoding="utf-8")
        cycle = _parse_text(resolve_file(folder / "a.yaml", output_format="json"), "json")
        assert cycle == {"other": {"back": None}}

    print("Jinest self-test: OK")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", nargs="?", help="Input .json/.yaml/.yml file")
    parser.add_argument("-o", "--output", help="Output file; stdout when omitted")
    parser.add_argument(
        "--output-format",
        choices=("json", "yaml"),
        help="Override output format",
    )
    parser.add_argument(
        "--unsafe",
        action="store_true",
        help="Disable the Jinja sandbox (trusted templates only)",
    )
    parser.add_argument(
        "-silent",
        "--no-messages",
        action="store_true",
        help="Do not print collected warnings and hints to stderr",
    )
    parser.add_argument(
        "-Werror",
        "--treat-warnings-as-errors",
        action="store_true",
        help="Abort when any warning is collected",
    )
    parser.add_argument("--self-test", action="store_true", help="Run built-in tests")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.input:
        parser.error("input is required unless --self-test is used")

    try:
        rendered = resolve_file(
            args.input,
            output=args.output,
            output_format=args.output_format,
            sandboxed=not args.unsafe,
            emit_messages=not args.no_messages,
            treat_warnings_as_errors=args.treat_warnings_as_errors,
        )
    except (JinestError, OSError, ValueError) as exc:
        parser.exit(1, f"jinest: {exc}\n")
    if args.output is None:
        print(rendered)


if __name__ == "__main__":
    _main()
