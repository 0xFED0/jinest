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
* ``name(args)=`` — declare a structural function with a lazy mapping or array
  body rebound at each call site.
* ``name+``, ``name~``, ``name*``, and ``name%`` — strict array transforms:
  one-level flatten, string join, Cartesian product, and equal-length zip.
* ``.name`` — the hidden channel, available to Jinja as ``name`` while a
  sibling public ``name`` remains independently materialized; ``name-: null``
  deletes public, ``.name-: null`` deletes hidden, and ``name.: null`` hides
  public output.
* A key ending in a backtick is a raw literal key; all Jinest key parsing is
  disabled and the marker is removed.
* ``=$expr``, ``=@text``, and ``=^script`` — inline scalar directives for
  native expressions, text templates, and scripts; they also form dynamic
  mapping keys when used as keys.
* ``<$``, ``<@``, ``<^``, ``<(args)=``, and ``<[axis=source]=`` — self-declaration
  wrappers that apply one declaration to the current value slot.
* Local priority is ``name`` > ``name^`` > ``name$`` > ``name@`` >
  ``name*`` > ``name+`` > ``name%`` > ``name~``.
* ``<<=`` / ``<<N=`` attach direct mapping default layers; ``<<!=`` /
  ``<<!N=`` are their override variants.
* ``<<$`` / ``<<N$`` and ``<<^`` / ``<<N^`` add evaluated default layers.
* ``<<!$`` / ``<<!N$`` and ``<<!^`` / ``<<!N^`` add evaluated override layers.
* ``<<[]`` / ``<<N[]`` and their ``!`` variants expand a list of lazy layers.
* Prefixing any layer form with ``.`` makes the values it contributes hidden.
* Lookup priority is last override, local, last default.
* ``context`` is the destination node, ``origin`` is the source declaration
  node, ``root`` is the source tree root, and ``global_root`` is the top-level
  Resolver root.
* ``path`` is an immutable PathRef for the destination context. Nodes expose
  ``path``, ``source_path``, ``root``, and ``file`` metadata attributes.
* Path helpers: ``normalize_path``, ``absolute_path``, ``relative_path``,
  ``path_of``, ``source_path_of``, ``at``, ``get``, ``root_of``,
  ``source_file``, and ``source_dir``.
* Lists are ordinary lazy nodes; explicit ``=$``, ``=@``, ``=^``, or self
  wrappers select evaluation for individual items.
* ``import_yaml`` (alias ``import``) and ``import_json`` load lazy trees whose
  source ``root`` and file metadata remain independent.
* ``Resolver.messages`` collects ``warning`` and ``hint`` diagnostics; use
  ``emit_messages=False`` or ``treat_warnings_as_errors=True`` to control them.

Requires: Jinja2 >= 3.1, < 4.
When this module is copied and used directly, PyYAML is optional and needed
only for YAML input/output. The published wheel installs PyYAML by default.
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
import weakref
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType, SimpleNamespace
from enum import Enum
from itertools import product
from dataclasses import dataclass, field, replace
from datetime import date, time
from pathlib import Path
from typing import Any, Iterator, MutableMapping, MutableSequence, NoReturn

from jinja2 import ChainableUndefined, StrictUndefined, Undefined, nodes, pass_context
from jinja2.exceptions import UndefinedError
from jinja2.compiler import CodeGenerator
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
    "PathRefInfo",
    "JinestFunction",
    "Resolver",
    "helpers",
    "resolve",
    "resolve_text",
    "resolve_file",
]

__version__ = "0.19.3"

_INTERNAL_SCOPE = "__jinest_scope__"
_INTERNAL_FUNCTION_LOCALS = "__jinest_function_locals__"
_INTERNAL_EVALUATOR_CONTEXT = "__jinest_evaluator_context__"
_INTERNAL_LEXICAL_VARS = "__jinest_lexical_vars__"
_RESERVED_NAMES = {
    # Context values are intrinsic to every Jinest evaluator. Stdlib globals
    # are intentionally *not* listed here: each Resolver derives them from its
    # enabled namespace selection.
    _INTERNAL_SCOPE,
    _INTERNAL_FUNCTION_LOCALS,
    _INTERNAL_EVALUATOR_CONTEXT,
    _INTERNAL_LEXICAL_VARS,
    "root",
    "global_root",
    "context",
    "origin",
    "_",
    "path",
    "keyname",
    "effective_key",
    "keymode",
    "keypath",
}
_MERGE_RE = re.compile(
    # ``!`` is always before the numeric order. ``[]`` changes source
    # multiplicity, while ``=`` accepts an already parsed mapping directly.
    r"^(?P<hidden>\.)?<<(?P<leading_override>!?)(?P<order>\d*)"
    r"(?P<mode>=|[$^]|\[\])$"
)
_INVALID_LEGACY_MERGE_RE = re.compile(r"^\.?<<\d+!(?:=|[$^]|\[\])$")
_MISSING = object()
_EMPTY_MAPPING: Mapping[Any, Any] = {}
_NODE_META_NAMES = {"path", "source_path", "root", "file"}


class JinestError(Exception):
    """Base error raised by Jinest."""

    def __init__(
        self,
        message: str = "",
        *,
        path: str | None = None,
        file: str | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.file = file


class JinestTemplateError(JinestError):
    """A Jinja expression or template could not be evaluated."""


class JinestFunctionError(JinestError):
    """A Jinest template function could not be called safely."""


@dataclass(frozen=True, slots=True)
class JinestMessage:
    """A diagnostic collected while resolving a Jinest tree.

    ``level`` is currently ``"warning"`` or ``"hint"`` and ``msg`` is the
    human-readable diagnostic text. ``path`` and ``file`` identify the source
    location when available. Message objects are intentionally small and
    immutable so callers can safely inspect or copy ``Resolver.messages``.
    """

    level: str
    msg: str
    path: str | None = None
    file: str | None = None


class JinestWarningError(JinestError):
    """Warnings were configured to abort resolution."""


class JinestMergeError(JinestError):
    """A merge directive did not produce a mapping."""


class JinestImportError(JinestError):
    """An imported JSON/YAML file could not be loaded."""


class JinestPathError(JinestError):
    """A Jinest path could not be parsed or resolved."""


@dataclass(slots=True, eq=False)
class _SourceDocument:
    """One source-root identity hosted by a Resolver runtime.

    Most resolvers host their ordinary parsed document. Synthetic nodes may
    host an additional ephemeral document without paying for another Resolver
    or registering it in the persistent DocumentStore.
    """

    raw: Any
    identity: object
    source_label: str | None = None
    root: "_ContainerProxy | None" = None


@dataclass(frozen=True, slots=True)
class _Source:
    """One raw container and its source-document location."""

    resolver: "Resolver"
    raw: Any
    source_path: tuple[Any, ...] = ()
    document: _SourceDocument | None = None

    @property
    def document_id(self) -> "_DocumentNodeId":
        """Stable identity of the declaration location in its document."""
        return _DocumentNodeId(
            (
                self.resolver._document_identity
                if self.document is None
                else self.document.identity
            ),
            self.source_path,
        )

    @property
    def instance_id(self) -> "_SourceInstanceId":
        """Identity of the concrete raw value at one declaration location.

        An evaluation may temporarily use an empty cycle sentinel and later
        produce a real mapping at the same source path. Runtime caches must
        never conflate those values.
        """
        return _SourceInstanceId(self.document_id, id(self.raw), self.raw)


@dataclass(frozen=True, slots=True)
class _DocumentNodeId:
    """Identity of one source-node occurrence, independent of raw aliases."""

    document_identity: object
    source_path: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _SourceInstanceId:
    """One concrete raw container at a document location."""

    document: _DocumentNodeId
    raw_id: int
    raw: Any = field(compare=False, hash=False, repr=False)


@dataclass(frozen=True, slots=True)
class _DeclarationId:
    """Identity of one declaration in a source node."""

    node: _DocumentNodeId
    source_key: Any


@dataclass(frozen=True, slots=True)
class _BindingId:
    """Identity of one destination attachment."""

    resolver_id: int
    serial: int


@dataclass(slots=True)
class _EvaluationFrame:
    local_vars: Mapping[str, Any] | None = None
    function_scope: "_ContainerProxy | None" = None
    function_origin_source: _Source | None = None
    function_body_source_path: tuple[Any, ...] | None = None
    context_origin_source: _Source | None = None


@dataclass(slots=True)
class _BindingCache:
    children: dict[Any, tuple[Any, Any]] = field(default_factory=dict)
    resolved: dict[Any, Any] = field(default_factory=dict)
    public_resolved: dict[Any, Any] = field(default_factory=dict)
    layers: dict[Any, "_LayerValue | None"] = field(default_factory=dict)
    normalized_layers: dict[tuple[_SourceInstanceId, int | None], "_LayerStack"] = field(
        default_factory=dict
    )
    key_indexes: dict[Any, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _Binding:
    """Destination state and mutable caches for one bound source node."""

    identity: _BindingId
    frame: _EvaluationFrame
    cache: _BindingCache = field(default_factory=_BindingCache)


@dataclass(frozen=True, slots=True)
class _LayerSpec:
    source_key: Any
    template: Any
    order: int
    position: int
    override: bool
    mode: EvaluatorKind | None
    direct: bool = False
    hidden: bool = False
    multiple: bool = False
    item_sequence: "_SequenceProxy | None" = None
    item_index: int | None = None


@dataclass(frozen=True, slots=True)
class _LayerValue:
    """A resolved layer source plus locals carried by structural results."""

    source: _Source
    local_vars: Mapping[str, Any] | None = None
    context_origin_source: _Source | None = None
    hidden: bool = False


@dataclass(slots=True)
class _LayerStack:
    """Partially visible destination-local layer topology during normalization."""

    defaults: list[_LayerSpec] = field(default_factory=list)
    overrides: list[_LayerSpec] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _Candidate:
    source_key: Any
    template: Any
    mode: str  # concrete, evaluator, compose, or function
    behavior: str = "value"  # value, delete, hide


@dataclass(frozen=True, slots=True)
class _CandidateLocation:
    """One field candidate together with its lazy source context."""

    candidate: _Candidate
    source: _Source
    local_vars: Mapping[str, Any] | None = None
    context_origin_source: _Source | None = None


@dataclass(frozen=True, slots=True)
class _FieldMatch:
    """The first effective VALUE candidate and its output visibility."""

    location: _CandidateLocation
    masked: bool


@dataclass(frozen=True, slots=True)
class _FunctionParameter:
    name: str
    default: str | None


@dataclass(frozen=True, slots=True)
class _FunctionSpec:
    name: str
    source_key: str
    template: Any
    mode: EvaluatorKind
    parameters: tuple[_FunctionParameter, ...]


@dataclass(frozen=True, slots=True)
class _ComposeAxis:
    name: str
    source: str


@dataclass(frozen=True, slots=True)
class _ComposeSpec:
    name: str
    source_key: str
    template: Any
    mode: EvaluatorKind
    axes: tuple[_ComposeAxis, ...]


def _compose_local_frame(
    inherited_locals: Mapping[str, Any] | None,
    spec: _ComposeSpec,
    axis_values: Sequence[Sequence[Any]],
    indexes: tuple[int, ...],
    combination_index0: int,
    combination_length: int,
) -> dict[str, Any]:
    """Build one compose frame, including immutable iteration metadata."""
    frame = dict(inherited_locals or {})
    values = tuple(
        axis_values[axis_index][position]
        for axis_index, position in enumerate(indexes)
    )
    frame.update(
        {axis.name: value for axis, value in zip(spec.axes, values)}
    )

    axis_metadata: dict[str, Mapping[str, Any]] = {}
    for axis, source_values, index in zip(spec.axes, axis_values, indexes):
        length = len(source_values)
        axis_metadata[axis.name] = MappingProxyType(
            {
                "index": index + 1,
                "index0": index,
                "length": length,
                "first": index == 0,
                "last": index == length - 1,
            }
        )
    frame["axis"] = MappingProxyType(axis_metadata)
    frame["axes"] = MappingProxyType(
        {
            "index": combination_index0 + 1,
            "index0": combination_index0,
            "length": combination_length,
            "first": combination_index0 == 0,
            "last": combination_index0 == combination_length - 1,
        }
    )
    return frame


@dataclass(frozen=True, slots=True)
class _SelfSpec:
    """One parsed declaration wrapper applied to its containing value slot."""

    source_key: str
    mode: str  # native, text, script, structural, compose_structural
    payload: Any  # expression body, _FunctionSpec, or _ComposeSpec


@dataclass(slots=True)
class CompiledMapping:
    """Syntax-only representation of one raw mapping source."""

    raw: Mapping[Any, Any]
    defaults: tuple[_LayerSpec, ...]
    overrides: tuple[_LayerSpec, ...]
    functions: tuple[_FunctionSpec, ...]
    composes: tuple[_ComposeSpec, ...]


class SyntaxCompiler:
    """Compile raw mappings once; destination-dependent keys stay unresolved."""

    def __init__(self, resolver: "Resolver") -> None:
        self._resolver = resolver
        self._cache: dict[int, CompiledMapping] = {}

    def compile(self, raw: Mapping[Any, Any]) -> CompiledMapping:
        raw_id = id(raw)
        cached = self._cache.get(raw_id)
        if cached is not None and cached.raw is raw:
            return cached
        compiled = self._resolver._compile_mapping(raw)
        self._cache[raw_id] = compiled
        return compiled

    def clear(self) -> None:
        self._cache.clear()


class EvaluatorKind(str, Enum):
    """Typed evaluator kinds; str compatibility preserves existing internals."""

    NATIVE = "native"
    TEXT = "text"
    SCRIPT = "script"
    STRUCTURAL = "structural"


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    """One evaluator application, regardless of field/inline/self syntax."""

    kind: EvaluatorKind
    template: Any
    source_key: Any


@dataclass(frozen=True, slots=True)
class LayerResult:
    """Value produced by an optional inner declaration layer."""

    value: Any
    applied: bool = False
    escaped_literal: bool = False


class EvaluatorRegistry:
    """Single dispatch point for scalar evaluator execution."""

    def apply(self, resolver: "Resolver", plan: EvaluationPlan, **kwargs: Any) -> Any:
        return resolver._render_plan(plan, **kwargs)


_FUNCTION_MODES = {
    "$": EvaluatorKind.NATIVE,
    "@": EvaluatorKind.TEXT,
    "^": EvaluatorKind.SCRIPT,
}
_FUNCTION_DECLARATION_MODES = {**_FUNCTION_MODES, "=": EvaluatorKind.STRUCTURAL}
_FUNCTION_MODE_MARKERS = {value: key for key, value in _FUNCTION_MODES.items()}
_ARRAY_TRANSFORM_MODES = {
    "+": "flatten",
    "~": "join",
    "*": "product",
    "%": "zip",
}
_ARRAY_TRANSFORM_MARKERS = {
    mode: marker for marker, mode in _ARRAY_TRANSFORM_MODES.items()
}
_FIELD_SUFFIX_MODES = (
    ("^", "script"),
    ("$", "native"),
    ("@", "text"),
    ("*", "product"),
    ("+", "flatten"),
    ("%", "zip"),
    ("~", "join"),
)


def _valid_evaluator_body(value: Any, mode: str) -> bool:
    """Whether a scalar can serve directly as one evaluator body."""
    if isinstance(value, str):
        return True
    return mode in {"native", "script"} and isinstance(value, (bool, int, float))


def _evaluator_body_requirement(mode: str) -> str:
    return (
        "a string body"
        if mode == "text"
        else "a string body or numeric/boolean scalar"
    )


@dataclass(frozen=True, slots=True)
class _MappingKeyEntry:
    """One source mapping key after raw/dynamic-key normalization."""

    source_key: Any
    key: Any
    raw: bool = False
    dynamic: bool = False
    compose: bool = False


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
    """Remove exactly one escape before an inline directive-shaped string."""
    if not isinstance(value, str) or not value.startswith("`"):
        return None
    index = 0
    while index < len(value) and value[index] == "`":
        index += 1
    if index + 1 < len(value) and value[index] == "=" and value[index + 1] in _FUNCTION_MODES:
        return value[1:]
    return None


def _self_syntax_key(key: Any) -> bool:
    """Whether a key is reserved for current-slot wrapper syntax.

    These keys stay literal unless their containing mapping is recognized as a
    one-key self wrapper.  Do not reserve arbitrary ``<...`` keys: they retain
    the ordinary Jinest key grammar.
    """
    if not isinstance(key, str) or _raw_key(key) is not None:
        return False
    return (
        key in {"<$", "<@", "<^"}
        or (key.startswith("<(") and key.endswith(")="))
        or (key.startswith("<[") and key.endswith("]="))
    )


def _literal_syntax_key(key: Any) -> bool:
    """Whether a source key must bypass every Jinest key parser."""
    return (
        _self_syntax_key(key)
        or _raw_key(key) is not None
        or _inline_directive(key) is not None
        or _escaped_inline_literal(key) is not None
    )


def _field_control_key(key: Any) -> tuple[str, str, str] | None:
    """Parse channel-specific DELETE/HIDE controls after key escaping."""
    if not isinstance(key, str) or _literal_syntax_key(key):
        return None
    if key.endswith("-"):
        if key.startswith("."):
            return key[1:-1], "hidden", "delete"
        return key[:-1], "public", "delete"
    if key.endswith(".") and not key.startswith("."):
        return key[:-1], "public", "hide"
    return None


def _parse_function_declaration(key: Any, template: Any = _MISSING) -> _FunctionSpec | None:
    """Parse a suffixed function declaration key without evaluating defaults."""
    if not isinstance(key, str) or _literal_syntax_key(key):
        return None
    marker = key[-1:] if key else ""
    if marker not in _FUNCTION_DECLARATION_MODES:
        if (
            template is not _MISSING
            and key.endswith(")")
            and isinstance(template, (Mapping, Sequence))
            and not isinstance(template, (str, bytes, bytearray))
        ):
            raise JinestError(
                f"Structural function declaration {key!r} requires a trailing '='"
            )
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

    mode = _FUNCTION_DECLARATION_MODES[marker]
    if template is not _MISSING:
        if mode == "structural" and not (
            isinstance(template, (Mapping, Sequence))
            and not isinstance(template, (str, bytes, bytearray))
        ):
            raise JinestError(
                f"Structural function {key!r} must have a mapping or array body"
            )
        if mode != "structural" and not _valid_evaluator_body(template, mode):
            raise JinestError(
                f"Function {key!r} requires {_evaluator_body_requirement(mode)}, "
                f"got {type(template).__name__}"
            )

    return _FunctionSpec(
        name=name,
        source_key=key,
        template=template,
        mode=mode,
        parameters=tuple(parameters),
    )


def _parse_compose_declaration(
    key: Any,
    template: Any = _MISSING,
) -> _ComposeSpec | None:
    """Parse ``name[axis=source, ...]=`` and ``...@`` declarations."""
    if not isinstance(key, str) or _literal_syntax_key(key):
        return None
    marker = key[-1:] if key else ""
    if marker not in {"=", "@"}:
        return None
    declaration = key[:-1]
    match = re.fullmatch(r"([A-Za-z_]\w*)\[(.*)\]", declaration, flags=re.DOTALL)
    if match is None:
        if "[" in declaration or "]" in declaration:
            raise JinestError(f"Malformed compose declaration {key!r}")
        return None
    name, axes_text = match.groups()
    if not axes_text.strip():
        raise JinestError(f"Compose declaration {key!r} requires at least one axis")
    source = f"def __jinest_compose__({axes_text}):\n    pass\n"
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise JinestError(
            f"Malformed compose declaration {key!r}: {exc.msg}"
        ) from exc
    function_node = tree.body[0]
    if not isinstance(function_node, ast.FunctionDef):
        raise JinestError(f"Malformed compose declaration {key!r}")
    arguments = function_node.args
    if (
        arguments.posonlyargs
        or arguments.vararg is not None
        or arguments.kwonlyargs
        or arguments.kwarg is not None
    ):
        raise JinestError(
            f"Unsupported axes in compose declaration {key!r}: "
            "*args, **kwargs, positional-only, and keyword-only axes are unsupported"
        )
    if any(argument.annotation is not None for argument in arguments.args):
        raise JinestError(
            f"Type annotations are unsupported in compose declaration {key!r}"
        )
    if len(arguments.defaults) != len(arguments.args):
        raise JinestError(
            f"Every compose axis in {key!r} must use name=source syntax"
        )
    axes: list[_ComposeAxis] = []
    for argument, default_node in zip(arguments.args, arguments.defaults):
        if argument.arg in {"axis", "axes"}:
            raise JinestError(
                f"Compose axis name {argument.arg!r} in {key!r} is reserved; "
                "use another name (axis and axes are service locals)"
            )
        axis_source = ast.get_source_segment(source, default_node)
        if axis_source is None:
            raise JinestError(
                f"Could not read source for compose axis {argument.arg!r} in {key!r}"
            )
        axes.append(_ComposeAxis(argument.arg, axis_source))
    mode = EvaluatorKind.STRUCTURAL if marker == "=" else EvaluatorKind.TEXT
    if template is not _MISSING:
        if mode == "structural" and not (
            isinstance(template, (Mapping, Sequence))
            and not isinstance(template, (str, bytes, bytearray))
        ):
            raise JinestError(
                f"Structural compose {key!r} must have a mapping or array body"
            )
        if mode == "text" and not isinstance(template, str):
            raise JinestError(f"Text compose {key!r} must have a string body")
    return _ComposeSpec(name, key, template, mode, tuple(axes))


def _parse_self_declaration(value: Any) -> _SelfSpec | None:
    """Parse a one-key ``<...`` wrapper without materializing its marker."""
    if not isinstance(value, Mapping) or len(value) != 1:
        return None
    # Do not call ``value.items()`` here: a Jinest mapping may legitimately
    # contain a field named ``items``, which shadows the Mapping method via
    # its lazy attribute lookup.  The protocol operations are unambiguous.
    key = next(iter(value))
    body = value[key]
    if not isinstance(key, str) or not key.startswith("<"):
        return None
    if key in {"<$", "<@", "<^"}:
        return _SelfSpec(
            key,
            {"<$": "native", "<@": "text", "<^": "script"}[key],
            body,
        )
    if key.startswith("<(") and key.endswith(")="):
        synthetic_key = "__jinest_self" + key[1:]
        try:
            function = _parse_function_declaration(synthetic_key, body)
        except JinestError as exc:
            raise JinestError(str(exc).replace(repr(synthetic_key), repr(key), 1)) from exc
        if function is None:
            raise JinestError(f"Malformed self structural function {key!r}")
        function = _FunctionSpec(
            name="__jinest_self",
            source_key=key,
            template=function.template,
            mode=function.mode,
            parameters=function.parameters,
        )
        return _SelfSpec(key, "structural", function)
    if key.startswith("<[") and key.endswith("]="):
        synthetic_key = "__jinest_self" + key[1:]
        try:
            compose = _parse_compose_declaration(synthetic_key, body)
        except JinestError as exc:
            raise JinestError(str(exc).replace(repr(synthetic_key), repr(key), 1)) from exc
        if compose is None:
            raise JinestError(f"Malformed self structural compose {key!r}")
        compose = _ComposeSpec(
            name="__jinest_self",
            source_key=key,
            template=compose.template,
            mode=compose.mode,
            axes=compose.axes,
        )
        return _SelfSpec(key, "compose_structural", compose)
    return None


class JinestFunction:
    """A lazy Jinest function declaration, callable from Jinja or Python."""

    __slots__ = ("_jinest_owner", "_jinest_spec", "_jinest_source", "_jinest_scope")

    def __init__(
        self,
        owner: "Resolver",
        spec: _FunctionSpec,
        source: _Source,
        scope: "_ContainerProxy | None" = None,
    ) -> None:
        object.__setattr__(self, "_jinest_owner", owner)
        object.__setattr__(self, "_jinest_spec", spec)
        object.__setattr__(self, "_jinest_source", source)
        object.__setattr__(self, "_jinest_scope", scope)

    def _python_call(
        self,
        scope: "_ContainerProxy",
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        vars: Mapping[str, Any] | None = None,
    ) -> Any:
        """Single Python-to-Jinest invocation bridge used by call() and fn()."""
        owner = object.__getattribute__(self, "_jinest_owner")
        source = object.__getattribute__(self, "_jinest_source")
        locals_map = dict(object.__getattribute__(scope, "_jinest_binding").frame.local_vars or {})
        if vars is not None:
            if not isinstance(vars, Mapping):
                raise TypeError("vars must be a mapping")
            locals_map.update(vars)
        return owner._invoke_python_function(self, scope, source, args, kwargs, locals_map)

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke this declaration immediately in the context that exposed it."""
        owner = object.__getattribute__(self, "_jinest_owner")
        scope = owner._coerce_destination_scope(
            default=object.__getattribute__(self, "_jinest_scope")
        )
        return self._python_call(scope, args, kwargs)

    def fn(
        self,
        *,
        context: Any = _MISSING,
        vars: Mapping[str, Any] | None = None,
    ) -> Any:
        """Return a Python callable adapter, optionally rebound with locals."""
        owner = object.__getattribute__(self, "_jinest_owner")
        spec = object.__getattribute__(self, "_jinest_spec")
        scope = owner._coerce_destination_scope(
            context, default=object.__getattribute__(self, "_jinest_scope")
        )
        if vars is not None and not isinstance(vars, Mapping):
            raise TypeError("vars must be a mapping")
        bound_vars = None if vars is None else MappingProxyType(dict(vars))

        def adapter(*args: Any, **kwargs: Any) -> Any:
            return self._python_call(scope, args, kwargs, bound_vars)

        adapter.__name__ = spec.name
        adapter.__qualname__ = spec.name
        return adapter

    @pass_context
    def __call__(self, jinja_context: Context, *args: Any, **kwargs: Any) -> Any:
        return object.__getattribute__(self, "_jinest_owner")._invoke_function(
            self, jinja_context, args, kwargs
        )

    def __repr__(self) -> str:
        spec = object.__getattribute__(self, "_jinest_spec")
        return f"<JinestFunction {spec.name}>"


@dataclass(frozen=True, slots=True)
class PathRefInfo:
    """Read-only Python metadata for a :class:`PathRef`."""

    owner: "Resolver"
    kind: str
    segments: tuple[Any, ...]
    relative: bool
    anchor: tuple[Any, ...]
    up: int


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

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        """Keep path references immutable after construction.

        Construction and internal derivation deliberately use
        :func:`object.__setattr__`; ordinary callers must never be able to
        mutate a path that may already be used as a cache key.
        """
        raise AttributeError("PathRef objects are immutable")

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_jinest_") or name.startswith("__"):
            return object.__getattribute__(self, name)
        if name == "_":
            return object.__getattribute__(self, "_jinest_parent")()
        if name == "absolute":
            return object.__getattribute__(self, "_jinest_absolute")()
        if name == "info":
            return PathRefInfo(
                object.__getattribute__(self, "_jinest_owner"),
                object.__getattribute__(self, "_jinest_kind"),
                object.__getattribute__(self, "_jinest_segments"),
                object.__getattribute__(self, "_jinest_relative"),
                object.__getattribute__(self, "_jinest_anchor_segments"),
                object.__getattribute__(self, "_jinest_up"),
            )
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

        # Function arguments and Jinja lexical locals are explicit variables.
        # They shadow attachments and fields, but unrelated globals must not
        # jump ahead of Jinest lookup merely because a local frame exists.
        # ``Context.resolve_or_missing()`` cannot express that distinction: it
        # searches both ``vars`` and ``parent`` (which also contains globals).
        value = self.vars.get(key, missing)
        if value is not missing:
            return value
        if function_locals is not None and key in function_locals:
            return function_locals[key]

        reserved_names = (
            object.__getattribute__(scope, "_jinest_owner")._reserved_names
            if isinstance(scope, _ContainerProxy)
            else _RESERVED_NAMES
        )
        if key not in reserved_names and isinstance(scope, _ContainerProxy):
            value = scope._jinest_resolve_name(key)
            if value is not _MISSING:
                return value

        value = super().resolve_or_missing(key)
        if value is not missing:
            return value
        return missing


class _JinestCodeGenerator(CodeGenerator):
    """Carry visible Jinja lexical variables into context-aware operations.

    Jinja normally derives a context with assignments made *inside* a loop,
    but the loop target, macro parameters and macro-local assignments only
    exist as generated Python locals. ``Symbols.dump_stores`` is the compiler
    abstraction for those locals. Passing them through Jinja's own
    ``_loop_vars`` channel keeps ordinary callable dispatch untouched, while a
    private keyword lets Jinest's filter/test adapters derive the same frame.

    Keeping this hook in ``signature`` avoids copying Jinja's sizeable
    ``visit_For`` implementation and therefore reduces version coupling.
    """

    @staticmethod
    def _jinest_lexical_vars(frame: Any) -> str | None:
        stores = frame.symbols.dump_stores()
        if not stores:
            return None
        return "{" + ", ".join(
            f"{name!r}: {reference}" for name, reference in stores.items()
        ) + "}"

    def signature(
        self,
        node: nodes.Call | nodes.Filter | nodes.Test,
        frame: Any,
        extra_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        additions = dict(extra_kwargs or {})
        lexical_vars = self._jinest_lexical_vars(frame)
        if isinstance(node, nodes.Call) and lexical_vars is not None:
            # Context.call consumes this internal argument before invoking the
            # callable, so normal Python/Jinja callables are unaffected.
            additions["_loop_vars"] = lexical_vars
        elif not isinstance(node, nodes.Call):
            surface = (
                self.environment.filters
                if isinstance(node, nodes.Filter)
                else self.environment.tests
            )
            function = surface.get(node.name)
            if getattr(function, "_jinest_contextual", False):
                additions[_INTERNAL_LEXICAL_VARS] = lexical_vars
        super().signature(node, frame, additions or None)


class _SandboxedNativeEnvironment(SandboxedEnvironment, NativeEnvironment):
    """NativeEnvironment with Jinest-specific sandbox checks."""

    def is_safe_attribute(self, obj: Any, attr: str, value: Any) -> bool:
        # ``PathRef.info`` is deliberately Python-only. PathRef otherwise keeps
        # attribute navigation semantics, so a real ``info`` path segment is
        # still reachable with brackets.
        if attr == "info" and isinstance(obj, PathRef):
            return False
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
        "_jinest_binding",
        "__weakref__",
    )

    def __init__(
        self,
        owner: "Resolver",
        source: _Source,
        parent: "_ContainerProxy | None",
        path: tuple[Any, ...],
        path_kind: str = "global",
        local_vars: Mapping[str, Any] | None = None,
        function_scope: "_ContainerProxy | None" = None,
        function_origin_source: _Source | None = None,
        function_body_source_path: tuple[Any, ...] | None = None,
        context_origin_source: _Source | None = None,
    ) -> None:
        object.__setattr__(self, "_jinest_owner", owner)
        object.__setattr__(self, "_jinest_source", source)
        object.__setattr__(self, "_jinest_parent", parent)
        object.__setattr__(self, "_jinest_path", path)
        object.__setattr__(self, "_jinest_path_kind", path_kind)
        object.__setattr__(self, "_jinest_children", {})
        frame = _EvaluationFrame(
            local_vars,
            function_scope,
            function_origin_source,
            function_body_source_path,
            context_origin_source,
        )
        binding = owner._new_binding(frame)
        object.__setattr__(self, "_jinest_binding", binding)
        object.__setattr__(self, "_jinest_children", binding.cache.children)
        owner._register_live_node(self)

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
                else owner._source_root_for(source)
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
                source.resolver._source_root_for(source),
                "source",
                source.source_path,
            )
        if name == "root":
            source = object.__getattribute__(self, "_jinest_source")
            return source.resolver._source_root_for(source)
        if name == "file":
            source = object.__getattribute__(self, "_jinest_source")
            return source.resolver._source_label_for(source)
        return object.__getattribute__(self, name)

    def __repr__(self) -> str:
        path = object.__getattribute__(self, "_jinest_path")
        kind = object.__getattribute__(self, "_jinest_path_kind")
        prefix = "global_root" if kind == "global" else "root"
        return f"<{type(self).__name__} path={_format_path_segments(prefix, path)}>"

    def __str__(self) -> str:
        owner = object.__getattribute__(self, "_jinest_owner")
        return str(owner._to_plain(self, state=_MaterializationState()))

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
        local_vars: Mapping[str, Any] | None = None,
        function_scope: "_ContainerProxy | None" = None,
        function_origin_source: _Source | None = None,
        function_body_source_path: tuple[Any, ...] | None = None,
        context_origin_source: _Source | None = None,
    ) -> None:
        super().__init__(
            owner,
            source,
            parent,
            path,
            path_kind,
            local_vars,
            function_scope,
            function_origin_source,
            function_body_source_path,
            context_origin_source,
        )
        cache = object.__getattribute__(self, "_jinest_binding").cache
        object.__setattr__(self, "_jinest_resolved", cache.resolved)
        object.__setattr__(self, "_jinest_public_resolved", cache.public_resolved)
        object.__setattr__(self, "_jinest_layer_cache", cache.layers)
        object.__setattr__(self, "_jinest_key_indexes", cache.key_indexes)

    def __getattribute__(self, name: str) -> Any:
        if name == "_" or name in _NODE_META_NAMES:
            return super().__getattribute__(name)

        if not name.startswith("_jinest_") and not name.startswith("__"):
            value = self._jinest_resolve_name(name)
            if value is not _MISSING:
                return value

        return object.__getattribute__(self, name)

    def _jinest_resolve_name(self, key: str) -> Any:
        owner = object.__getattribute__(self, "_jinest_owner")
        if owner._scope_has_logical(self, key):
            return owner._get_field(self, key)
        function_scope = object.__getattribute__(self, "_jinest_binding").frame.function_scope
        if isinstance(function_scope, _ContainerProxy) and function_scope is not self:
            return function_scope._jinest_resolve_name(key)
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
    """Lazy sequence whose items may contain inline or self declarations."""

    __slots__ = (
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
        key_context: tuple[Any, Any] | None = None,
        path_kind: str = "global",
        local_vars: Mapping[str, Any] | None = None,
        function_scope: "_ContainerProxy | None" = None,
        function_origin_source: _Source | None = None,
        function_body_source_path: tuple[Any, ...] | None = None,
        context_origin_source: _Source | None = None,
    ) -> None:
        super().__init__(
            owner,
            source,
            parent,
            path,
            path_kind,
            local_vars,
            function_scope,
            function_origin_source,
            function_body_source_path,
            context_origin_source,
        )
        object.__setattr__(self, "_jinest_key_context", key_context)
        cache = object.__getattribute__(self, "_jinest_binding").cache
        object.__setattr__(self, "_jinest_resolved", cache.resolved)

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
            item_path = object.__getattribute__(self, "_jinest_path") + (normalized,)

            key_context = object.__getattribute__(self, "_jinest_key_context")
            keyname = None if key_context is None else key_context[0]
            effective_key = None if key_context is None else key_context[1]
            layer_result = owner._resolve_layer_input(
                self,
                item,
                origin_source=source,
                source_key=normalized,
                context_path=item_path,
                keyname=keyname,
                effective_key=effective_key,
            )
            if not layer_result.applied and not layer_result.escaped_literal:
                value = item
            else:
                value = layer_result.value
            if owner._is_container(value):
                value = owner._bind_child(
                    self,
                    normalized,
                    value,
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


@dataclass(frozen=True, slots=True)
class ResolverConfig:
    """Validated immutable Resolver options shared by imported documents."""

    in_place: bool
    strict: bool
    sandboxed: bool
    globals: Mapping[str, Any]
    filters: Mapping[str, Any]
    source_path: Path | None
    base_dir: Path
    import_roots: tuple[Path, ...] | None
    function_max_depth: int
    emit_messages: bool
    treat_warnings_as_errors: bool
    debug: bool
    stdlib: frozenset[str]
    stdlib_exclude: frozenset[str]


@dataclass(slots=True)
class DiagnosticSink:
    """Shared ordered diagnostics for one global resolution tree."""

    messages: list[JinestMessage] = field(default_factory=list)
    keys: set[tuple[Any, ...]] = field(default_factory=set)
    emitted_count: int = 0

    def add(self, message: JinestMessage, key: tuple[Any, ...]) -> bool:
        if key in self.keys:
            return False
        self.keys.add(key)
        self.messages.append(message)
        return True

    def clear(self) -> None:
        """Discard diagnostics from a completed in-place resolution run."""
        self.messages.clear()
        self.keys.clear()
        self.emitted_count = 0


@dataclass(frozen=True, slots=True)
class _ImportedDocument:
    """Parsed filesystem import reused by independent lazy resolvers."""

    data: Any
    identity: tuple[str, Path, str]


@dataclass(frozen=True, slots=True)
class _TreeDocument:
    """Snapshotted in-memory source shared by ancestry-specific runtimes."""

    data: Any
    identity: str
    source_path: Path | None
    base_dir: Path


@dataclass(slots=True)
class DocumentStore:
    """Parsed import cache scoped to one top-level resolution run."""

    cache: dict[tuple[Path, str], _ImportedDocument] = field(default_factory=dict)
    runtimes: dict[tuple[Path, str, tuple[Path, ...]], "Resolver"] = field(
        default_factory=dict
    )
    tree_documents: dict[str, _TreeDocument] = field(default_factory=dict)
    tree_runtimes: dict[
        tuple[str, tuple[str, ...], tuple[Path, ...]], "Resolver"
    ] = field(default_factory=dict)

    def get(self, path: Path, format: str) -> _ImportedDocument | None:
        return self.cache.get((path, format))

    def put(self, path: Path, format: str, data: Any) -> _ImportedDocument:
        document = _ImportedDocument(data, ("import", path, format))
        self.cache[(path, format)] = document
        return document

    def clear(self) -> None:
        self.cache.clear()
        self.runtimes.clear()
        self.tree_documents.clear()
        self.tree_runtimes.clear()


@dataclass(slots=True)
class _MaterializationState:
    """Active identities while converting lazy nodes to plain Python values."""

    bindings: set[_BindingId] = field(default_factory=set)
    # A physical raw object may occur at several paths in one document. Its
    # active occurrence is a cycle, but the same object in an independent
    # source document is not. This guard is therefore deliberately narrower
    # than declaration identity: document identity plus raw object identity.
    proxy_raw: dict[tuple[object, int], list["_ContainerProxy"]] = field(
        default_factory=dict
    )
    plain_raw: set[int] = field(default_factory=set)


@dataclass(slots=True)
class Materializer:
    """Plain-value materialization boundary for one Resolver."""

    resolver: "Resolver"

    def materialize(self, value: Any) -> Any:
        return self.resolver._to_plain(value, state=_MaterializationState())


@dataclass(slots=True)
class JinjaBridge:
    """Jinja environments and compiled-artifact cache for one Resolver."""

    environment: Any
    script_environment: Any
    compilation_cache: dict[tuple[_DeclarationId, EvaluatorKind, str], Any] = field(
        default_factory=dict
    )
    template_cache: dict[tuple[EvaluatorKind, str], Any] = field(default_factory=dict)


class SerializationCodecs:
    """Input/output codec boundary used by the public helper functions."""

    @staticmethod
    def parse(text: str, format: str) -> Any:
        return _parse_text(text, format)

    @staticmethod
    def serialize(value: Any, format: str) -> str:
        return _serialize(value, format)



@dataclass(frozen=True, slots=True)
class _StdlibExports:
    """Explicit Jinja export surfaces for one stdlib namespace."""

    globals: Mapping[str, Any] = field(default_factory=dict)
    filters: Mapping[str, Any] = field(default_factory=dict)
    tests: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _StdlibCallFrame:
    """The current Jinest/Jinja call frame shared by stdlib adapters."""

    resolver: "Resolver"
    context: Context
    scope: "_ContainerProxy"
    origin: "_ContainerProxy"
    root: "_ContainerProxy"
    local_vars: Mapping[str, Any]

    @classmethod
    def from_jinja(cls, context: Context) -> "_StdlibCallFrame":
        scope = context.vars.get(_INTERNAL_SCOPE, context.parent.get(_INTERNAL_SCOPE))
        if not isinstance(scope, _ContainerProxy):
            raise JinestTemplateError("Jinest stdlib is unavailable outside a Jinest evaluation frame")
        resolver = object.__getattribute__(scope, "_jinest_owner")
        origin = context.resolve_or_missing("origin")
        if origin is missing or not isinstance(origin, _ContainerProxy):
            origin = object.__getattribute__(scope, "_jinest_source").resolver._source_view(
                object.__getattribute__(scope, "_jinest_source")
            )
        root = context.resolve_or_missing("root")
        if root is missing or not isinstance(root, _ContainerProxy):
            root = origin.root
        # A derived Context places loop locals in ``parent`` rather than
        # ``vars``. Start with every visible value, then remove the exact
        # environment-global bindings; keeping those would incorrectly shadow
        # ordinary Jinest fields in a nested eval. The remaining values are
        # lexical ``set``/loop bindings plus Jinest's own call-frame locals.
        local_vars = dict(object.__getattribute__(scope, "_jinest_binding").frame.local_vars or {})
        environment_globals = context.environment.globals
        base_context = context.vars.get(
            _INTERNAL_EVALUATOR_CONTEXT,
            context.parent.get(_INTERNAL_EVALUATOR_CONTEXT, {}),
        )
        for name, value in context.get_all().items():
            if name in {
                _INTERNAL_SCOPE,
                _INTERNAL_FUNCTION_LOCALS,
                _INTERNAL_EVALUATOR_CONTEXT,
            }:
                continue
            # Do not feed unmodified service variables back as locals: doing
            # so would override an explicit context/origin/root on the nested
            # runtime call. A lexical set/loop binding remains visible because
            # it either lives in vars or differs from the base evaluator value.
            if (
                name in _RESERVED_NAMES
                and name not in context.vars
                and base_context.get(name, _MISSING) is value
            ):
                continue
            if name in environment_globals and value is environment_globals[name]:
                continue
            local_vars[name] = value
        return cls(resolver, context, scope, origin, root, MappingProxyType(local_vars))

    def call(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        """Invoke one template callable through Jinja's sandbox call path."""
        return self.context.environment.call(self.context, function, *args, **kwargs)

    def _scope(self, context: Any = _MISSING) -> "_ContainerProxy":
        return self.resolver._coerce_destination_scope(context, default=self.scope)

    def _origin_source(self, origin: Any = _MISSING, root: Any = _MISSING) -> _Source:
        origin_node = self.resolver._coerce_api_scope(origin, default=self.origin)
        source = object.__getattribute__(origin_node, "_jinest_source")
        if root is not _MISSING:
            root_node = self.resolver._coerce_api_scope(root)
            root_source = object.__getattribute__(root_node, "_jinest_source")
            if not self.resolver._same_source_tree(root_source, source):
                raise JinestError("origin and root must belong to the same source tree")
        return source

    def merged_vars(self, vars: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        if vars is None:
            return self.local_vars
        if not isinstance(vars, Mapping):
            raise TypeError("vars must be a mapping")
        result = dict(self.local_vars)
        result.update(vars)
        return result

    def render(
        self,
        template: Any,
        mode: str,
        *,
        context: Any = _MISSING,
        origin: Any = _MISSING,
        root: Any = _MISSING,
        vars: Mapping[str, Any] | None = None,
    ) -> Any:
        scope = self._scope(context)
        source = self._origin_source(origin, root)
        return self.resolver._render(
            scope,
            template,
            mode=mode,
            origin_source=source,
            source_key=None,
            local_vars=self.merged_vars(vars),
            context_origin_source=source,
        )


class _StdlibNamespace:
    """Base class for declarative, side-effect-free stdlib namespaces."""

    name: str

    def __init__(self, resolver: "Resolver") -> None:
        self.resolver = resolver

    def exports(self) -> _StdlibExports:
        raise NotImplementedError

    def contextual(self, handler: Any) -> Any:
        @pass_context
        def wrapper(context: Context, *args: Any, **kwargs: Any) -> Any:
            lexical_vars = kwargs.pop(_INTERNAL_LEXICAL_VARS, None)
            if lexical_vars:
                context = context.derived(lexical_vars)
            return handler(_StdlibCallFrame.from_jinja(context), *args, **kwargs)
        wrapper._jinest_contextual = True
        return wrapper


class _PathStdlib(_StdlibNamespace):
    name = "path"

    def exports(self) -> _StdlibExports:
        def normalize(frame: _StdlibCallFrame, value: Any, anchor: Any = _MISSING) -> PathRef:
            return frame.resolver._normalize_path(value, frame=frame.context, anchor=anchor)
        def absolute(frame: _StdlibCallFrame, value: Any, anchor: Any = _MISSING, **kwargs: Any) -> PathRef:
            if "root" in kwargs and anchor is _MISSING:
                anchor = kwargs.pop("root")
            if kwargs:
                raise TypeError(f"Unexpected arguments: {', '.join(kwargs)}")
            return frame.resolver._absolute_path(value, frame=frame.context, anchor=anchor)
        def relative(frame: _StdlibCallFrame, target: Any, base: Any = _MISSING, **kwargs: Any) -> PathRef:
            if "path" in kwargs and base is _MISSING:
                base = kwargs.pop("path")
            if kwargs:
                raise TypeError(f"Unexpected arguments: {', '.join(kwargs)}")
            return frame.resolver._relative_path(target, frame=frame.context, base=base)
        def at(frame: _StdlibCallFrame, target: Any, anchor: Any = _MISSING) -> Any:
            return frame.resolver._at(target, anchor=anchor, frame=frame.context)
        def get(frame: _StdlibCallFrame, target: Any, default: Any = None, anchor: Any = _MISSING) -> Any:
            try:
                return at(frame, target, anchor)
            except (KeyError, IndexError, TypeError, JinestPathError, UndefinedError):
                return default
        def path_of(frame: _StdlibCallFrame, value: Any) -> PathRef:
            return _api_path_of(value)
        def source_path_of(frame: _StdlibCallFrame, value: Any) -> PathRef:
            return _api_path_of(value, source=True)
        def root_of(frame: _StdlibCallFrame, value: Any) -> Any:
            return _api_root_of(value)
        def source_file(frame: _StdlibCallFrame, value: Any) -> str | None:
            return _api_source_file(value)
        def source_dir(frame: _StdlibCallFrame, value: Any = _MISSING) -> str | None:
            return _api_source_dir(frame.origin if value is _MISSING else value)
        handlers = {
            "normalize_path": normalize, "absolute_path": absolute,
            "relative_path": relative, "path_of": path_of,
            "source_path_of": source_path_of, "at": at, "get": get,
            "root_of": root_of, "source_file": source_file, "source_dir": source_dir,
        }
        wrapped = {name: self.contextual(handler) for name, handler in handlers.items()}
        return _StdlibExports(wrapped, wrapped)


class _FilesStdlib(_StdlibNamespace):
    name = "files"

    def exports(self) -> _StdlibExports:
        def file_path(frame: _StdlibCallFrame, path: Any, anchor: Any = _MISSING) -> str:
            # Never expose pathlib.Path as a template capability: its public
            # read/write methods would bypass the read-only stdlib boundary and
            # could traverse outside import_roots after this policy check.
            return str(
                _api_file_path(
                    path,
                    anchor=frame.origin if anchor is _MISSING else anchor,
                )
            )
        def read_text(frame: _StdlibCallFrame, path: Any, encoding: str = "utf-8", anchor: Any = _MISSING) -> str:
            return _api_read_text(path, encoding=encoding, anchor=frame.origin if anchor is _MISSING else anchor)
        def read_lines(frame: _StdlibCallFrame, path: Any, encoding: str = "utf-8", keepends: bool = False, anchor: Any = _MISSING) -> list[str]:
            return _api_read_lines(
                path,
                encoding=encoding,
                keepends=keepends,
                anchor=frame.origin if anchor is _MISSING else anchor,
            )
        def read_bytes(frame: _StdlibCallFrame, path: Any, anchor: Any = _MISSING) -> bytes:
            return _api_read_bytes(path, anchor=frame.origin if anchor is _MISSING else anchor)
        def file_exists(frame: _StdlibCallFrame, path: Any, anchor: Any = _MISSING) -> bool:
            return _api_file_path(path, anchor=frame.origin if anchor is _MISSING else anchor).exists()
        handlers = {"file_path": file_path, "read_text": read_text, "read_lines": read_lines, "read_bytes": read_bytes, "file_exists": file_exists}
        wrapped = {name: self.contextual(handler) for name, handler in handlers.items()}
        return _StdlibExports(wrapped, wrapped)


class _RuntimeStdlib(_StdlibNamespace):
    name = "runtime"

    def exports(self) -> _StdlibExports:
        def node(frame: _StdlibCallFrame, value: Any, context: Any = _MISSING, origin: Any = _MISSING, root: Any = _MISSING, vars: Mapping[str, Any] | None = None) -> Any:
            if isinstance(value, _ContainerProxy):
                if context is origin is root is _MISSING and vars is None:
                    return value
                # Rebinding an existing node changes its destination attachment,
                # not its source document.  Leave origin/root unset unless the
                # caller explicitly requested an override; otherwise a node
                # returned by import_tree/import_json would be forced into the
                # caller's unrelated source tree.
                return frame.resolver.node(
                    value,
                    context=frame.scope if context is _MISSING else context,
                    origin=origin,
                    root=root,
                    vars=frame.merged_vars(vars),
                    _lookup_scope=frame.scope,
                )
            return frame.resolver.node(
                value,
                context=frame.scope if context is _MISSING else context,
                origin=frame.origin if origin is _MISSING else origin,
                root=frame.root if root is _MISSING else root,
                vars=frame.merged_vars(vars),
                _lookup_scope=frame.scope,
            )
        def resolve(frame: _StdlibCallFrame, value: Any, context: Any = _MISSING, origin: Any = _MISSING, root: Any = _MISSING, vars: Mapping[str, Any] | None = None) -> Any:
            if isinstance(value, (PathRef, _ContainerProxy)) and context is origin is root is _MISSING and vars is None:
                return _api_owner(value).resolve(value)
            if isinstance(value, PathRef):
                value = _api_owner(value)._at_path(value)
            if not frame.resolver._is_container(value):
                if any(item is not _MISSING for item in (context, origin, root)) or vars is not None:
                    raise TypeError("binding overrides require a mapping/list/node target")
                return value
            node_value = node(frame, value, context=context, origin=origin, root=root, vars=vars)
            return _api_owner(node_value).resolve(node_value)
        def dynamic(mode: str) -> Any:
            def call(frame: _StdlibCallFrame, source: Any, context: Any = _MISSING, origin: Any = _MISSING, root: Any = _MISSING, vars: Mapping[str, Any] | None = None) -> Any:
                return frame.render(source, mode, context=context, origin=origin, root=root, vars=vars)
            return call
        handlers = {
            "node": node, "resolve": resolve, "eval": dynamic("native"),
            "render": dynamic("text"), "script": dynamic("script"),
            "literal": lambda frame, value: _literal_tree(value),
        }
        wrapped = {name: self.contextual(handler) for name, handler in handlers.items()}
        return _StdlibExports(wrapped, wrapped)


class _DocumentsStdlib(_StdlibNamespace):
    name = "documents"

    def exports(self) -> _StdlibExports:
        def load(frame: _StdlibCallFrame, path: Any, format: str, encoding: str = "utf-8", anchor: Any = _MISSING) -> Any:
            return _api_load_document(
                path,
                format,
                encoding=encoding,
                anchor=frame.origin if anchor is _MISSING else anchor,
            )
        def import_document(frame: _StdlibCallFrame, path: Any, format: str, anchor: Any = _MISSING) -> Any:
            return _api_import(path, format, anchor=frame.origin if anchor is _MISSING else anchor)
        handlers = {
            "load_json": lambda frame, path, encoding="utf-8", anchor=_MISSING: load(frame, path, "json", encoding, anchor),
            "load_yaml": lambda frame, path, encoding="utf-8", anchor=_MISSING: load(frame, path, "yaml", encoding, anchor),
            "import_json": lambda frame, path, anchor=_MISSING: import_document(frame, path, "json", anchor),
            "import_yaml": lambda frame, path, anchor=_MISSING: import_document(frame, path, "yaml", anchor),
            "import_tree": lambda frame, value, source, base_dir=None: _api_owner(
                frame.origin
            ).import_tree(value, source=source, base_dir=base_dir),
        }
        wrapped = {name: self.contextual(handler) for name, handler in handlers.items()}
        globals_map = dict(wrapped)
        globals_map["import"] = wrapped["import_yaml"]
        # ``import`` as a filter predates namespace registration. Keep this
        # documented compatibility alias while all new exports stay explicit.
        filters_map = dict(wrapped)
        filters_map["import"] = wrapped["import_yaml"]
        return _StdlibExports(globals_map, filters_map)


class _SerializationStdlib(_StdlibNamespace):
    name = "serialization"

    def exports(self) -> _StdlibExports:
        def materialize(frame: _StdlibCallFrame, value: Any) -> Any:
            return _api_materialize(value)
        handlers = {
            "from_json": lambda frame, text: SerializationCodecs.parse(text, "json"),
            "from_yaml": lambda frame, text: SerializationCodecs.parse(text, "yaml"),
            "to_json": lambda frame, value: SerializationCodecs.serialize(materialize(frame, value), "json"),
            "to_yaml": lambda frame, value: SerializationCodecs.serialize(materialize(frame, value), "yaml"),
            "json_normalize": lambda frame, value: _normalize_json_value(materialize(frame, value), active=set()),
            "yaml_normalize": lambda frame, value: _normalize_yaml_value(materialize(frame, value), active=set()),
        }
        wrapped = {name: self.contextual(handler) for name, handler in handlers.items()}
        return _StdlibExports(wrapped, wrapped)


def _stdlib_flatten(value: Any, levels: int | None = None) -> list[Any]:
    if levels is not None and (
        not isinstance(levels, int) or isinstance(levels, bool) or levels < 0
    ):
        raise TypeError("flatten levels must be a non-negative integer or None")

    def is_nested(item: Any) -> bool:
        return isinstance(item, Iterable) and not isinstance(
            item, (str, bytes, bytearray, Mapping)
        )

    result: list[Any] = []
    active: set[int] = set()

    def append(item: Any, depth: int | None) -> None:
        if not is_nested(item) or depth == 0:
            result.append(item)
            return
        identity = id(item)
        if identity in active:
            raise JinestError("Cyclic iterable cannot be flattened")
        active.add(identity)
        try:
            next_depth = None if depth is None else depth - 1
            for nested in item:
                append(nested, next_depth)
        finally:
            active.remove(identity)

    if is_nested(value):
        identity = id(value)
        active.add(identity)
        try:
            for item in value:
                append(item, levels)
        finally:
            active.remove(identity)
    else:
        result.append(value)
    return result


def _stdlib_zip(*values: Any, strict: bool = False) -> list[list[Any]]:
    if not values:
        return []
    iterators = [iter(value) for value in values]
    result: list[list[Any]] = []
    while True:
        row: list[Any] = []
        exhausted = 0
        for iterator in iterators:
            try:
                row.append(next(iterator))
            except StopIteration:
                exhausted += 1
        if exhausted:
            if strict and exhausted != len(iterators):
                raise JinestError("zip(strict=True) requires iterables of equal length")
            return result
        result.append(row)


def _stdlib_product(*values: Any) -> list[list[Any]]:
    return [list(row) for row in product(*values)]


class _CollectionsStdlib(_StdlibNamespace):
    name = "collections"

    def exports(self) -> _StdlibExports:
        def enumerate_values(frame: _StdlibCallFrame, values: Any, start: int = 0) -> list[list[Any]]:
            return [[index, value] for index, value in enumerate(values, start)]
        def map_values(frame: _StdlibCallFrame, function: Any, *iterables: Any) -> list[Any]:
            if not iterables:
                raise TypeError("map() requires at least one iterable")
            return [frame.call(function, *values) for values in zip(*iterables)]
        def filter_values(frame: _StdlibCallFrame, predicate: Any, values: Any) -> list[Any]:
            return [value for value in values if frame.call(predicate, value)]
        def apply(frame: _StdlibCallFrame, value: Any, function: Any, *args: Any, **kwargs: Any) -> Any:
            return frame.call(function, value, *args, **kwargs)
        def any_values(frame: _StdlibCallFrame, values: Any, predicate: Any = _MISSING) -> bool:
            for value in values:
                candidate = value if predicate is _MISSING else frame.call(predicate, value)
                if candidate:
                    return True
            return False
        def all_values(frame: _StdlibCallFrame, values: Any, predicate: Any = _MISSING) -> bool:
            for value in values:
                candidate = value if predicate is _MISSING else frame.call(predicate, value)
                if not candidate:
                    return False
            return True
        handlers = {
            "flatten": lambda frame, value, levels=None: _stdlib_flatten(value, levels),
            "zip": lambda frame, *values, strict=False: _stdlib_zip(*values, strict=strict),
            "enumerate": enumerate_values,
            "product": lambda frame, *values: _stdlib_product(*values),
            "combine": lambda frame, first, second, recursive=False: _combine(first, second, recursive),
            "apply": apply,
            "any": any_values, "all": all_values,
            "union": lambda frame, *values: _ordered_union(*values),
            "intersect": lambda frame, first, *rest: _ordered_intersect(first, *rest),
            "difference": lambda frame, first, *rest: _ordered_filter(first, _ordered_union(*rest), include=False),
            "symmetric_difference": lambda frame, first, second: _ordered_union(_ordered_filter(first, second, include=False), _ordered_filter(second, first, include=False)),
        }
        wrapped = {name: self.contextual(handler) for name, handler in handlers.items()}
        globals_map = dict(wrapped)
        globals_map["map"] = self.contextual(map_values)
        globals_map["filter"] = self.contextual(filter_values)
        return _StdlibExports(globals_map, wrapped)


class _MathStdlib(_StdlibNamespace):
    name = "math"

    def exports(self) -> _StdlibExports:
        def clamp(frame: _StdlibCallFrame, value: Any, minimum: Any, maximum: Any) -> Any:
            if minimum > maximum:
                raise JinestError("clamp minimum must not exceed maximum")
            return min(max(value, minimum), maximum)
        handlers = {
            "clamp": clamp, "ceil": lambda frame, value: math.ceil(value),
            "floor": lambda frame, value: math.floor(value), "sqrt": lambda frame, value: math.sqrt(value),
            "log": lambda frame, value, base=None: math.log(value) if base is None else math.log(value, base),
        }
        wrapped = {name: self.contextual(handler) for name, handler in handlers.items()}
        return _StdlibExports(wrapped, wrapped)


def _regex_flags(ignorecase: bool = False, multiline: bool = False, dotall: bool = False) -> re.RegexFlag:
    # ``re.NOFLAG`` was added in Python 3.11.  The zero-valued RegexFlag
    # is the same neutral value and keeps Jinest's documented 3.10 support.
    flags = re.RegexFlag(0)
    if ignorecase:
        flags |= re.IGNORECASE
    if multiline:
        flags |= re.MULTILINE
    if dotall:
        flags |= re.DOTALL
    return flags


def _regex_compile(pattern: Any, **kwargs: Any) -> re.Pattern[str]:
    try:
        return re.compile(str(pattern), _regex_flags(**kwargs))
    except re.error as exc:
        raise JinestTemplateError(f"Invalid regular expression {pattern!r}: {exc}") from exc


class _StringsStdlib(_StdlibNamespace):
    name = "strings"

    def exports(self) -> _StdlibExports:
        def regex(operation: str) -> Any:
            def call(frame: _StdlibCallFrame, value: Any, pattern: Any, **kwargs: Any) -> Any:
                compiled = _regex_compile(pattern, **kwargs)
                return bool(getattr(compiled, operation)(str(value)))
            return call
        def findall(frame: _StdlibCallFrame, value: Any, pattern: Any, **kwargs: Any) -> list[Any]:
            return _regex_compile(pattern, **kwargs).findall(str(value))
        def replace(frame: _StdlibCallFrame, value: Any, pattern: Any, replacement: Any, count: int = 0, **kwargs: Any) -> str:
            return _regex_compile(pattern, **kwargs).sub(str(replacement), str(value), count=count)
        def split(frame: _StdlibCallFrame, value: Any, pattern: Any, maxsplit: int = 0, **kwargs: Any) -> list[str]:
            return _regex_compile(pattern, **kwargs).split(str(value), maxsplit=maxsplit)
        handlers = {
            "split": lambda frame, value, sep=None, maxsplit=-1: str(value).split(sep, maxsplit),
            "regex_match": regex("match"), "regex_fullmatch": regex("fullmatch"),
            "regex_search": regex("search"), "regex_findall": findall,
            "regex_replace": replace, "regex_split": split,
            "regex_escape": lambda frame, value: re.escape(str(value)),
        }
        wrapped = {name: self.contextual(handler) for name, handler in handlers.items()}
        tests = {name: wrapped[name] for name in ("regex_match", "regex_fullmatch", "regex_search")}
        return _StdlibExports(wrapped, wrapped, tests)


class _StdlibRegistry:
    """Collect and install deterministic Jinest Jinja stdlib exports."""

    _types = (_PathStdlib, _FilesStdlib, _RuntimeStdlib, _DocumentsStdlib,
              _SerializationStdlib, _CollectionsStdlib, _MathStdlib, _StringsStdlib)
    names = tuple(namespace.name for namespace in _types)

    def __init__(self, resolver: "Resolver", selection: bool | Sequence[str] = True, exclude: Sequence[str] = ()) -> None:
        available = set(self.names)

        def normalize_names(value: Any, option: str) -> set[str]:
            if isinstance(value, (str, bytes)):
                raise TypeError(f"{option} must be a collection of namespace names")
            try:
                names = set(value)
            except TypeError as exc:
                raise TypeError(f"{option} must be a collection of namespace names") from exc
            invalid = [name for name in names if not isinstance(name, str)]
            if invalid:
                rendered = ", ".join(repr(name) for name in invalid)
                raise TypeError(f"{option} namespace names must be strings, got {rendered}")
            unknown = names - available
            if unknown:
                raise ValueError(
                    f"Unknown Jinest stdlib namespace(s): {', '.join(sorted(unknown))}"
                )
            return names

        if isinstance(selection, bool):
            selected = set(available) if selection else set()
        else:
            selected = normalize_names(selection, "stdlib")
        excluded = normalize_names(exclude, "stdlib_exclude")
        selected -= excluded
        self.enabled = tuple(name for name in self.names if name in selected)
        self.excluded = frozenset(excluded)
        globals_map: dict[str, Any] = {}
        filters_map: dict[str, Any] = {}
        tests_map: dict[str, Any] = {}
        for namespace_type in self._types:
            if namespace_type.name not in selected:
                continue
            exports = namespace_type(resolver).exports()
            self._merge(globals_map, exports.globals, "global", namespace_type.name)
            self._merge(filters_map, exports.filters, "filter", namespace_type.name)
            self._merge(tests_map, exports.tests, "test", namespace_type.name)
        self.globals = MappingProxyType(globals_map)
        self.filters = MappingProxyType(filters_map)
        self.tests = MappingProxyType(tests_map)

    @staticmethod
    def _merge(destination: dict[str, Any], exports: Mapping[str, Any], surface: str, namespace: str) -> None:
        for name, value in exports.items():
            if name in destination:
                raise JinestError(f"Jinest stdlib {surface} collision for {name!r} while installing {namespace!r}")
            destination[name] = value

    def install(self, *environments: Any) -> None:
        for environment in environments:
            for surface, exports, existing in (
                ("global", self.globals, environment.globals),
                ("filter", self.filters, environment.filters),
                ("test", self.tests, environment.tests),
            ):
                collisions = set(exports) & set(existing)
                if collisions:
                    raise JinestError(
                        f"Jinest stdlib {surface} collides with standard Jinja name(s): "
                        f"{', '.join(sorted(collisions))}"
                    )
            environment.globals.update(self.globals)
            environment.filters.update(self.filters)
            environment.tests.update(self.tests)


class Resolver:
    """Resolve a structured Python tree containing lazy Jinja fields.

    Instances are stateful and not reentrant or thread-safe. Use one Resolver
    per concurrent resolution.
    """

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
        debug: bool = False,
        stdlib: bool | Sequence[str] = True,
        stdlib_exclude: Sequence[str] = (),
        _import_chain: tuple[Path, ...] | None = None,
        _tree_import_chain: tuple[str, ...] = (),
        _global_owner: "Resolver | None" = None,
        _documents: DocumentStore | None = None,
        _document_identity: object | None = None,
        _copy_input: bool = True,
        _source_label: str | None = None,
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
        if not isinstance(debug, bool):
            raise TypeError("debug must be a boolean")
        self.emit_messages = emit_messages
        self.treat_warnings_as_errors = treat_warnings_as_errors
        self.debug = debug
        self._global_owner = _global_owner or self
        if _document_identity is None:
            self._document_identity: object = ("memory", id(self))
        else:
            self._document_identity = _document_identity
        self._binding_serial = 0
        if self._global_owner is self:
            self._diagnostics = DiagnosticSink()
        else:
            self._diagnostics = self._global_owner._diagnostics
        self.messages = self._diagnostics.messages
        self._message_keys = self._diagnostics.keys
        self._function_depth = 0
        self._function_stack: list[str] = []
        self._user_globals = dict(globals or {})
        self._user_filters = dict(filters or {})
        self.stdlib = _StdlibRegistry(self, stdlib, stdlib_exclude)
        self._reserved_names = frozenset(_RESERVED_NAMES | set(self.stdlib.globals))
        self._original = data
        self._in_place_snapshot: Any = _MISSING
        if in_place:
            self.data = data
        elif not _copy_input:
            # Imported documents are immutable parsed payloads owned by the
            # per-run DocumentStore. Bindings never mutate their source, so
            # independent import occurrences can safely share this tree.
            self.data = data
        else:
            try:
                self.data = copy.deepcopy(data)
            except Exception:
                # Resolution never mutates its source tree. Falling back to the
                # original preserves strict=False semantics for unsupported
                # objects whose custom deepcopy implementation fails; the
                # scalar boundary below will still convert/reject them.
                self.data = data
        self._syntax = SyntaxCompiler(self)
        self._evaluators = EvaluatorRegistry()
        self._documents = _documents or DocumentStore()
        self._import_cache = self._documents.cache
        self._materializer = Materializer(self)
        self._source_view_cache: dict[_SourceInstanceId, _ContainerProxy] = {}
        # Weak references keep externally-held synthetic/rebound nodes in the
        # same invalidation graph without making their lifetime global.
        self._live_nodes: dict[int, weakref.ReferenceType[_ContainerProxy]] = {}
        self.source_path = Path(source_path).expanduser().resolve() if source_path else None
        self._source_label = _source_label or (str(self.source_path) if self.source_path else None)
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

        self.config = ResolverConfig(
            in_place,
            strict,
            sandboxed,
            MappingProxyType(self._user_globals.copy()),
            MappingProxyType(self._user_filters.copy()),
            self.source_path,
            self.base_dir,
            self.import_roots,
            function_max_depth,
            emit_messages,
            treat_warnings_as_errors,
            debug,
            frozenset(self.stdlib.enabled),
            self.stdlib.excluded,
        )

        if _import_chain is not None:
            self._import_chain = _import_chain
        elif self.source_path is not None:
            self._import_chain = (self.source_path,)
        else:
            self._import_chain = ()
        self._tree_import_chain = _tree_import_chain

        environment_type = _SandboxedNativeEnvironment if sandboxed else NativeEnvironment
        undefined_type = StrictUndefined if strict else ChainableUndefined
        self.environment = environment_type(undefined=undefined_type)
        self.script_environment = environment_type(
            undefined=undefined_type,
            extensions=[_ReturnExtension, 'jinja2.ext.do'],
            line_statement_prefix="%",
        )
        self.environment.context_class = _JinestContext
        self.script_environment.context_class = _JinestContext
        self.environment.code_generator_class = _JinestCodeGenerator
        self.script_environment.code_generator_class = _JinestCodeGenerator
        self._jinja = JinjaBridge(self.environment, self.script_environment)
        self._jinja_compilation_cache = self._jinja.compilation_cache

        # The registry owns all Jinest-added Jinja exports. User additions are
        # intentionally applied afterwards, so they can deliberately override
        # a selected stdlib name without rebuilding the registry.
        self.stdlib.install(self.environment, self.script_environment)
        self.environment.globals.update(self._user_globals)
        self.environment.filters.update(self._user_filters)
        self.script_environment.globals.update(self._user_globals)
        self.script_environment.filters.update(self._user_filters)

        try:
            self._reset_root_views()
        except Exception as exc:
            self._annotate_error(
                exc,
                path="root",
                file=str(self.source_path) if self.source_path else None,
            )
            self._emit_debug_error(exc)
            raise

    def _source_location(
        self,
        source: _Source | None = None,
        path: tuple[Any, ...] | None = None,
    ) -> tuple[str | None, str | None]:
        if source is None:
            return None, None
        source_path = source.source_path if path is None else path
        return (
            _format_path_segments("root", source_path),
            self._source_label_for(source),
        )

    def _record_message(
        self,
        level: str,
        msg: str,
        *,
        dedupe_key: tuple[Any, ...] | None = None,
        source: _Source | None = None,
        path: tuple[Any, ...] | None = None,
    ) -> None:
        """Add one deduplicated diagnostic to the shared resolver message list."""
        if level not in {"warning", "hint"}:
            raise ValueError(f"Unsupported message level: {level!r}")
        key = dedupe_key or (level, msg)
        message_path, message_file = self._source_location(source, path=path)
        self._diagnostics.add(
            JinestMessage(level, msg, message_path, message_file), key
        )

    def _debug_lines(self, *, path: str | None, file: str | None) -> list[str]:
        if not self.debug:
            return []
        return [f"  at {path or 'root'}", f"  in {file or '<memory>'}"]

    def _annotate_error(
        self,
        error: BaseException,
        *,
        path: str | None = None,
        file: str | None = None,
    ) -> BaseException:
        if isinstance(error, JinestError):
            if getattr(error, "path", None) is None:
                error.path = path
            if getattr(error, "file", None) is None:
                error.file = file
        return error

    def _emit_debug_error(self, error: BaseException) -> None:
        if not self.debug or getattr(error, "_jinest_debug_emitted", False):
            return
        setattr(error, "_jinest_debug_emitted", True)
        print(f"jinest: {error}", file=sys.stderr)
        for line in self._debug_lines(
            path=getattr(error, "path", None), file=getattr(error, "file", None)
        ):
            print(line, file=sys.stderr)

    def _flush_messages(self) -> None:
        """Print newly collected diagnostics, if stderr output is enabled."""
        owner = self._global_owner
        if not owner.emit_messages:
            owner._diagnostics.emitted_count = len(owner.messages)
            return
        pending = owner.messages[owner._diagnostics.emitted_count:]
        for message in pending:
            print(f"jinest: {message.level}: {message.msg}", file=sys.stderr)
            for line in owner._debug_lines(path=message.path, file=message.file):
                print(line, file=sys.stderr)
        owner._diagnostics.emitted_count = len(owner.messages)

    def _finalize_messages(self) -> None:
        """Apply warning policy and emit diagnostics after successful resolution."""
        owner = self._global_owner
        warnings = [message for message in owner.messages if message.level == "warning"]
        if owner.treat_warnings_as_errors and warnings:
            details = "\n".join(message.msg for message in warnings)
            error = JinestWarningError(
                f"Warnings treated as errors ({len(warnings)}):\n{details}"
            )
            first = warnings[0]
            error.path, error.file = first.path, first.file
            owner._emit_debug_error(error)
            raise error
        self._flush_messages()

    @staticmethod
    def _source_label_for(source: _Source) -> str | None:
        if source.document is not None:
            return source.document.source_label
        return source.resolver._source_label

    @staticmethod
    def _source_tree_identity(source: _Source) -> object:
        if source.document is not None:
            return source.document.identity
        return source.resolver._document_identity

    @classmethod
    def _same_source_tree(cls, left: _Source, right: _Source) -> bool:
        return cls._source_tree_identity(left) == cls._source_tree_identity(right)



    @staticmethod
    def _source_root_for(source: _Source) -> "_ContainerProxy":
        """Return the root of the exact document containing source."""
        document = source.document
        if document is None:
            root = source.resolver._source_root
        else:
            root = document.root
            if root is None:
                root = source.resolver._wrap(
                    document.raw,
                    parent=None,
                    path=(),
                    origin=source.resolver,
                    source_path=(),
                    source_document=document,
                    path_kind="source",
                )
                document.root = root
        if not isinstance(root, _ContainerProxy):
            raise JinestPathError("Source root must be a mapping or sequence")
        return root

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
            self._source_view_cache[root_source.instance_id] = self._source_root
        else:
            # Scalars have no template scope or source-view metadata.
            self._source_root = self.root
        self.global_root = self._global_owner.root
        if isinstance(self.data, Mapping):
            # Make root diagnostics available immediately through ``messages``;
            # nested mappings are discovered when their lazy scopes are visited.
            self._schema_for_source(root_source)

    def _reset_in_place_state(self, *, fresh_resolution: bool) -> None:
        """Rebuild reusable-root state without retaining prior resolution data."""
        if fresh_resolution:
            self._documents.clear()
            self._diagnostics.clear()
            self._function_depth = 0
            self._function_stack.clear()
        self._syntax.clear()
        self._jinja_compilation_cache.clear()
        self._jinja.template_cache.clear()
        self._source_view_cache.clear()
        self._reset_root_views()

    @staticmethod
    def _in_place_fingerprint(value: Any, *, active: set[int] | None = None) -> Any:
        """Return a comparison-safe snapshot of a materialized value.

        ``in_place`` accepts arbitrary ``MutableMapping`` and
        ``MutableSequence`` implementations. Their ``__eq__`` and
        ``__deepcopy__`` methods are application code, so neither may decide
        whether a resolved result is still current. The fingerprint contains
        only builtin immutable values and returns ``_MISSING`` for a value
        outside Jinest's materialized data model.
        """
        if active is None:
            active = set()
        if value is None:
            return ("none",)
        if isinstance(value, bool):
            return ("bool", value)
        if isinstance(value, int):
            return ("int", int(value))
        if isinstance(value, float):
            return ("float", float(value))
        if isinstance(value, str):
            return ("str", str(value))
        if isinstance(value, bytes):
            return ("bytes", bytes(value))
        if isinstance(value, bytearray):
            return ("bytearray", bytes(value))
        if isinstance(value, time):
            return ("time", type(value).__qualname__, value.isoformat())
        if isinstance(value, date):
            return ("date", type(value).__qualname__, value.isoformat())
        if isinstance(value, Mapping):
            value_id = id(value)
            if value_id in active:
                return _MISSING
            active.add(value_id)
            try:
                items: list[tuple[Any, Any]] = []
                for key, item in value.items():
                    frozen_key = Resolver._in_place_fingerprint(key, active=active)
                    frozen_item = Resolver._in_place_fingerprint(item, active=active)
                    if frozen_key is _MISSING or frozen_item is _MISSING:
                        return _MISSING
                    items.append((frozen_key, frozen_item))
                return ("mapping", tuple(items))
            except Exception:
                return _MISSING
            finally:
                active.remove(value_id)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            value_id = id(value)
            if value_id in active:
                return _MISSING
            active.add(value_id)
            try:
                items = tuple(
                    Resolver._in_place_fingerprint(item, active=active)
                    for item in value
                )
                if any(item is _MISSING for item in items):
                    return _MISSING
                return ("sequence", items)
            except Exception:
                return _MISSING
            finally:
                active.remove(value_id)
        return _MISSING

    def _commit_in_place_result(self, result: Any) -> None:
        """Replace the mutable root after all fallible resolution work succeeds."""
        if isinstance(self._original, MutableMapping) and isinstance(result, Mapping):
            previous = list(self._original.items())
            try:
                self._original.clear()
                self._original.update(result)
            except Exception:
                # Best-effort rollback for custom mutable mappings. Standard
                # dict/list operations cannot fail here, while custom classes
                # may reject one of the replacement operations.
                try:
                    self._original.clear()
                    self._original.update(previous)
                except Exception:
                    pass
                raise
            self.data = self._original
            return
        if (
            isinstance(self._original, MutableSequence)
            and not isinstance(self._original, (str, bytes, bytearray))
            and isinstance(result, list)
        ):
            previous = list(self._original)
            try:
                self._original[:] = result
            except Exception:
                try:
                    self._original[:] = previous
                except Exception:
                    pass
                raise
            self.data = self._original
            return
        self.data = result

    def resolve(
        self,
        target: Any = _MISSING,
        *,
        context: Any = _MISSING,
        origin: Any = _MISSING,
        root: Any = _MISSING,
        vars: Mapping[str, Any] | None = None,
    ) -> Any:
        """Materialize the root or one explicitly selected lazy value.

        Supplying ``target`` never commits an ``in_place`` resolver. Plain
        mappings/lists are first bound through :meth:`node`, so targeted and
        synthetic resolution share the ordinary lazy runtime.
        """
        if target is not _MISSING:
            value = self._coerce_api_target(
                target, context=context, origin=origin, root=root, vars=vars
            )
            try:
                result = self._materializer.materialize(value)
            except Exception as exc:
                self._flush_messages()
                self._emit_debug_error(exc)
                raise
            self._finalize_messages()
            return result
        if any(value is not _MISSING for value in (context, origin, root)) or vars is not None:
            raise TypeError("context, origin, root, and vars require an explicit target")
        if self.in_place:
            if self._in_place_snapshot is not _MISSING:
                if self._in_place_fingerprint(self.data) == self._in_place_snapshot:
                    return self.data
            self._reset_in_place_state(fresh_resolution=True)
        try:
            result = self._materializer.materialize(self.root)
        except Exception as exc:
            self._flush_messages()
            self._emit_debug_error(exc)
            raise
        self._finalize_messages()
        if self.in_place:
            snapshot = self._in_place_fingerprint(result)
            if snapshot is _MISSING:
                raise JinestError("Could not snapshot materialized in-place result")
            self._commit_in_place_result(result)
            self._in_place_snapshot = snapshot
            self.root = self.data
            self._source_root = self.data
            self.global_root = self.data
            return self.data
        return result

    def _coerce_api_scope(
        self, context: Any = _MISSING, *, default: _ContainerProxy | None = None
    ) -> _ContainerProxy:
        if context is _MISSING:
            context = default if default is not None else self.root
        if isinstance(context, PathRef):
            context = object.__getattribute__(context, "_jinest_owner")._at_path(context)
        elif isinstance(context, str):
            context = self._at(context, frame=None)
        if not isinstance(context, _ContainerProxy):
            raise TypeError("context must resolve to a Jinest mapping or list node")
        return context

    def _coerce_destination_scope(
        self, context: Any = _MISSING, *, default: _ContainerProxy | None = None
    ) -> _ContainerProxy:
        scope = self._coerce_api_scope(context, default=default)
        owner = object.__getattribute__(scope, "_jinest_owner")
        if owner._global_owner is not self._global_owner:
            raise JinestError("context must belong to the same Jinest resolver tree")
        return scope

    def _api_locals(self, scope: _ContainerProxy, vars: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if vars is None:
            return object.__getattribute__(scope, "_jinest_binding").frame.local_vars
        if not isinstance(vars, Mapping):
            raise TypeError("vars must be a mapping")
        result = dict(object.__getattribute__(scope, "_jinest_binding").frame.local_vars or {})
        result.update(vars)
        return result

    def _coerce_api_target(
        self, target: Any, *, context: Any, origin: Any, root: Any,
        vars: Mapping[str, Any] | None,
    ) -> Any:
        if isinstance(target, PathRef):
            owner = object.__getattribute__(target, "_jinest_owner")
            if owner._global_owner is not self._global_owner:
                raise JinestError(
                    "target must belong to the same Jinest resolver tree"
                )
            target = owner._at_path(target)
        if isinstance(target, _ContainerProxy):
            owner = object.__getattribute__(target, "_jinest_owner")
            if owner._global_owner is not self._global_owner:
                raise JinestError(
                    "target must belong to the same Jinest resolver tree"
                )
            if context is origin is root is _MISSING and vars is None:
                return target
            return self.node(target, context=context, origin=origin, root=root, vars=vars)
        if isinstance(target, Mapping) or (
            isinstance(target, Sequence) and not isinstance(target, (str, bytes, bytearray))
        ):
            return self.node(target, context=context, origin=origin, root=root, vars=vars)
        if any(value is not _MISSING for value in (context, origin, root)) or vars is not None:
            raise TypeError("binding overrides require a mapping/list/node target")
        return target

    def node(
        self, value: Any, *, context: Any = _MISSING, origin: Any = _MISSING,
        root: Any = _MISSING, vars: Mapping[str, Any] | None = None,
        _lookup_scope: _ContainerProxy | None = None,
    ) -> _ContainerProxy:
        """Bind a mapping/list as a lazy Jinest node without materializing it."""
        if not self._is_container(value):
            raise TypeError("node() requires a mapping or non-string sequence")
        if isinstance(value, _ContainerProxy):
            owner = object.__getattribute__(value, "_jinest_owner")
            if owner._global_owner is not self._global_owner:
                raise JinestError(
                    "value must belong to the same Jinest resolver tree"
                )
        if (
            isinstance(value, _ContainerProxy)
            and context is origin is root is _MISSING
            and vars is None
        ):
            return value

        existing_node = isinstance(value, _ContainerProxy)
        default_parent = (
            object.__getattribute__(value, "_jinest_parent")
            if existing_node else None
        )
        parent = self._coerce_destination_scope(context, default=default_parent)
        locals_map = self._api_locals(parent, vars)

        if existing_node:
            source = object.__getattribute__(value, "_jinest_source")
            source_origin = source.resolver
            source_path = source.source_path
            source_document = source.document
            context_origin_source = object.__getattribute__(
                value, "_jinest_binding"
            ).frame.context_origin_source
        else:
            source_origin = self
            source_path = ("<synthetic>", id(value))
            source_document = None
            context_origin_source = None

        if origin is not _MISSING:
            origin_node = self._coerce_api_scope(origin)
            context_origin_source = object.__getattribute__(
                origin_node, "_jinest_source"
            )
            source_origin = context_origin_source.resolver

        if root is not _MISSING:
            root_node = self._coerce_api_scope(root)
            root_source = object.__getattribute__(root_node, "_jinest_source")
            # A plain node may explicitly adopt a source root. Existing nodes
            # retain their own source and only accept a compatible override.
            if origin is _MISSING and not existing_node:
                source_origin = root_source.resolver
                context_origin_source = root_source
            elif not self._same_source_tree(
                context_origin_source if context_origin_source is not None else source,
                root_source,
            ):
                raise JinestError("origin and root must belong to the same source tree")

        if not existing_node and origin is _MISSING and root is _MISSING:
            # Host an independent, ephemeral source document in this runtime.
            # It is intentionally absent from DocumentStore: global_root and
            # all environment/import services still belong to this Resolver.
            source_document = _SourceDocument(value, object())
            source_path = ()

        path = object.__getattribute__(parent, "_jinest_path") + ("<node>",)
        bound = self._wrap(
            value,
            parent=parent,
            path=path,
            origin=source_origin,
            source_path=source_path,
            source_document=source_document,
            path_kind=object.__getattribute__(parent, "_jinest_path_kind"),
            local_vars=locals_map,
            function_scope=_lookup_scope,
            context_origin_source=context_origin_source,
        )
        if not isinstance(bound, _ContainerProxy):
            raise TypeError("node() requires a mapping or non-string sequence")
        return bound

    def _clear_binding_cache(self, node: _ContainerProxy, visited: set[_BindingId]) -> None:
        binding = object.__getattribute__(node, "_jinest_binding")
        if binding.identity in visited:
            return
        visited.add(binding.identity)
        cache = binding.cache
        for cached in tuple(cache.children.values()):
            child = cached[1]
            if isinstance(child, _ContainerProxy):
                self._clear_binding_cache(child, visited)
        cache.children.clear(); cache.resolved.clear(); cache.public_resolved.clear()
        cache.layers.clear(); cache.normalized_layers.clear(); cache.key_indexes.clear()

    def _runtime_graph(self) -> tuple["Resolver", ...]:
        """Return this global runtime and its cached independent documents."""
        owner = self._global_owner
        candidates = [owner, *owner._documents.runtimes.values(), *owner._documents.tree_runtimes.values()]
        result: list[Resolver] = []
        seen: set[int] = set()
        for candidate in candidates:
            if candidate._global_owner is owner and id(candidate) not in seen:
                seen.add(id(candidate))
                result.append(candidate)
        return tuple(result)

    def clear_cache(self, target: Any = _MISSING) -> None:
        """Invalidate lazy binding caches without discarding sources or globals."""
        if target is _MISSING:
            # Parsed documents and compiled templates deliberately survive;
            # only destination-local lazy state is invalidated.
            for runtime in self._runtime_graph():
                visited: set[_BindingId] = set()
                if isinstance(runtime.root, _ContainerProxy):
                    runtime._clear_binding_cache(runtime.root, visited)
                # Synthetic nodes and explicit rebinding can remain live in
                # Python without being attached under ``root``. They share
                # this resolver's runtime contract and must not stay stale.
                for reference in tuple(runtime._live_nodes.values()):
                    node = reference()
                    if node is not None:
                        runtime._clear_binding_cache(node, visited)
            return
        if isinstance(target, PathRef):
            owner = object.__getattribute__(target, "_jinest_owner")
            if owner._global_owner is not self._global_owner:
                raise JinestError(
                    "target must belong to the same Jinest resolver tree"
                )
            absolute = target._jinest_absolute()
            segments = object.__getattribute__(absolute, "_jinest_segments")
            if not segments:
                owner.clear_cache(object.__getattribute__(absolute, "_jinest_root")); return
            parent_path = PathRef(owner, object.__getattribute__(absolute, "_jinest_root"), object.__getattribute__(absolute, "_jinest_kind"), segments[:-1])
            parent = owner._at_path(parent_path)
            if isinstance(parent, _ContainerProxy):
                key = segments[-1]
                cache = object.__getattribute__(parent, "_jinest_binding").cache
                child = cache.children.get(key)
                if child is not None and isinstance(child[1], _ContainerProxy):
                    owner._clear_binding_cache(child[1], set())
                cache.children.pop(key, None); cache.resolved.pop(key, None); cache.public_resolved.pop(key, None)
                cache.key_indexes.clear(); cache.layers.clear(); cache.normalized_layers.clear()
                return
            raise JinestPathError(f"Path {target} has no container parent")
        if not isinstance(target, _ContainerProxy):
            raise TypeError("clear_cache() target must be a Jinest node or PathRef")
        owner = object.__getattribute__(target, "_jinest_owner")
        if owner._global_owner is not self._global_owner:
            raise JinestError("target must belong to the same Jinest resolver tree")
        owner._clear_binding_cache(target, set())

    def _apply_globals(self, mapping: Mapping[str, Any]) -> None:
        self._user_globals.update(mapping)
        for environment in (self.environment, self.script_environment):
            environment.globals.update(mapping)
        self.config = replace(self.config, globals=MappingProxyType(self._user_globals.copy()))

    def update_globals(self, mapping: Mapping[str, Any]) -> None:
        """Update globals for future evaluations throughout this resolver graph."""
        if not isinstance(mapping, Mapping):
            raise TypeError("globals must be a mapping")
        for runtime in self._runtime_graph():
            runtime._apply_globals(mapping)

    def _apply_filters(self, mapping: Mapping[str, Any]) -> None:
        self._user_filters.update(mapping)
        for environment in (self.environment, self.script_environment):
            environment.filters.update(mapping)
        # Jinja compiles filter lookup and pass-context calling conventions into
        # each artifact, and it may constant-fold a filter applied to literals.
        # Keep lazy field values intact, but discard only compiled artifacts so
        # future uncached evaluations observe the updated filter contract.
        self._jinja.compilation_cache.clear()
        self._jinja.template_cache.clear()
        self.config = replace(
            self.config,
            filters=MappingProxyType(self._user_filters.copy()),
        )

    def update_filters(self, mapping: Mapping[str, Any]) -> None:
        """Update filters for future evaluations throughout this resolver graph."""
        if not isinstance(mapping, Mapping):
            raise TypeError("filters must be a mapping")
        for runtime in self._runtime_graph():
            runtime._apply_filters(mapping)

    def _api_render(self, template: Any, mode: str, *, context: Any = _MISSING, vars: Mapping[str, Any] | None = None) -> Any:
        scope = self._coerce_destination_scope(context)
        source = object.__getattribute__(scope, "_jinest_source")
        return self._render(scope, template, mode=mode, origin_source=source,
                            source_key=None, local_vars=self._api_locals(scope, vars))

    def eval(self, expression: Any, *, context: Any = _MISSING, vars: Mapping[str, Any] | None = None) -> Any:
        return self._api_render(expression, "native", context=context, vars=vars)

    def render(self, template: Any, *, context: Any = _MISSING, vars: Mapping[str, Any] | None = None) -> str:
        return self._api_render(template, "text", context=context, vars=vars)

    def script(self, source: Any, *, context: Any = _MISSING, vars: Mapping[str, Any] | None = None) -> Any:
        return self._api_render(source, "script", context=context, vars=vars)

    def normalize_path(self, value: Any, *, anchor: Any = _MISSING) -> PathRef:
        return self._normalize_path(value, frame=None, anchor=anchor)

    def absolute_path(self, value: Any, *, anchor: Any = _MISSING) -> PathRef:
        return self._absolute_path(value, anchor=anchor, frame=None)

    def relative_path(self, target: Any, *, base: Any = _MISSING) -> PathRef:
        return self._relative_path(target, base=base, frame=None)

    def at(self, target: Any, *, anchor: Any = _MISSING) -> Any:
        return self._at(target, anchor=anchor, frame=None)

    def get(self, target: Any, default: Any = None, *, anchor: Any = _MISSING) -> Any:
        try:
            return self.at(target, anchor=anchor)
        except (KeyError, IndexError, TypeError, JinestPathError, UndefinedError):
            return default

    def import_tree(
        self,
        value: Any,
        *,
        source: str | os.PathLike[str],
        base_dir: str | os.PathLike[str] | None = None,
    ) -> Any:
        """Create/reuse an independent lazy document from a Python tree."""
        if not self._is_container(value):
            raise TypeError("import_tree() requires a mapping or non-string sequence")
        if isinstance(value, _ContainerProxy):
            # A document imports declarations, never a destination binding or
            # its caches/context. The new source identity is authoritative.
            value = object.__getattribute__(value, "_jinest_source").raw
        try:
            source_text = os.fspath(source)
        except TypeError as exc:
            raise TypeError("import_tree source must be a string or path-like value") from exc
        if not isinstance(source_text, str):
            raise TypeError("import_tree source must be a string or path-like value")
        if not source_text:
            raise ValueError("import_tree source must not be empty")

        is_virtual = re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", source_text)
        source_path: Path | None = None
        identity = source_text
        if not is_virtual:
            candidate = Path(source_text).expanduser()
            if (
                isinstance(source, os.PathLike)
                or candidate.is_absolute()
                or candidate.suffix.lower() in {".json", ".yaml", ".yml"}
            ):
                source_path = candidate.resolve()
                identity = str(source_path)

        if identity in self._tree_import_chain:
            return None
        tree_chain = self._tree_import_chain + (identity,)

        document = self._documents.tree_documents.get(identity)
        if document is None:
            try:
                snapshot = copy.deepcopy(value)
            except Exception as exc:
                raise JinestImportError(
                    f"Could not snapshot import_tree source {source_text!r}"
                ) from exc
            resolved_base = Path(
                base_dir
                if base_dir is not None
                else source_path.parent if source_path is not None else self.base_dir
            ).expanduser().resolve()
            document = _TreeDocument(
                snapshot,
                identity,
                source_path,
                resolved_base,
            )
            self._documents.tree_documents[identity] = document

        runtime_key = (identity, tree_chain, self._import_chain)
        cached = self._documents.tree_runtimes.get(runtime_key)
        if cached is not None:
            return cached.root
        child = Resolver(
            document.data,
            strict=self.strict,
            sandboxed=self.sandboxed,
            globals=self._user_globals,
            filters=self._user_filters,
            source_path=document.source_path,
            base_dir=document.base_dir,
            import_roots=self.import_roots,
            function_max_depth=self.function_max_depth,
            emit_messages=False,
            treat_warnings_as_errors=False,
            debug=self.debug,
            stdlib=self.config.stdlib,
            stdlib_exclude=self.config.stdlib_exclude,
            _import_chain=self._import_chain,
            _tree_import_chain=tree_chain,
            _global_owner=self._global_owner,
            _documents=self._documents,
            _document_identity=("tree", identity),
            _copy_input=False,
            _source_label=identity,
        )
        self._documents.tree_runtimes[runtime_key] = child
        return child.root


    # ------------------------------------------------------------------
    # Binding and source ownership
    # ------------------------------------------------------------------

    def _register_live_node(self, node: _ContainerProxy) -> None:
        """Track a node weakly so full cache invalidation reaches it too."""
        key = id(node)

        def discard(reference: weakref.ReferenceType[_ContainerProxy]) -> None:
            if self._live_nodes.get(key) is reference:
                self._live_nodes.pop(key, None)

        self._live_nodes[key] = weakref.ref(node, discard)

    def _new_binding(self, frame: _EvaluationFrame) -> _Binding:
        """Create destination-local state; no cache survives a rebind."""
        self._binding_serial += 1
        return _Binding(
            _BindingId(id(self), self._binding_serial),
            frame,
        )

    def _wrap(
        self,
        value: Any,
        parent: _ContainerProxy | None,
        path: tuple[Any, ...],
        *,
        origin: "Resolver | None" = None,
        source_path: tuple[Any, ...] | None = None,
        source_document: _SourceDocument | None = None,
        sequence_key_context: tuple[Any, Any] | None = None,
        path_kind: str = "global",
        local_vars: Mapping[str, Any] | None = None,
        function_scope: _ContainerProxy | None = None,
        function_origin_source: _Source | None = None,
        function_body_source_path: tuple[Any, ...] | None = None,
        context_origin_source: _Source | None = None,
    ) -> Any:
        if isinstance(value, _ContainerProxy):
            source = object.__getattribute__(value, "_jinest_source")
            if local_vars is None:
                local_vars = object.__getattribute__(value, "_jinest_binding").frame.local_vars
            if function_scope is None:
                function_scope = object.__getattribute__(value, "_jinest_binding").frame.function_scope
            if function_origin_source is None:
                function_origin_source = object.__getattribute__(value, "_jinest_binding").frame.function_origin_source
            if function_body_source_path is None:
                function_body_source_path = object.__getattribute__(value, "_jinest_binding").frame.function_body_source_path
            if context_origin_source is None:
                context_origin_source = object.__getattribute__(value, "_jinest_binding").frame.context_origin_source
        elif isinstance(value, Mapping):
            source = _Source(origin or self, value, tuple(source_path or ()), source_document)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            source = _Source(origin or self, value, tuple(source_path or ()), source_document)
        else:
            return self._copy_scalar(value)

        if isinstance(source.raw, Mapping):
            return _MappingProxy(
                self,
                source,
                parent,
                path,
                path_kind,
                local_vars,
                function_scope,
                function_origin_source,
                function_body_source_path,
                context_origin_source,
            )
        if isinstance(source.raw, Sequence) and not isinstance(
            source.raw, (str, bytes, bytearray)
        ):
            if isinstance(value, _SequenceProxy):
                if sequence_key_context is None:
                    sequence_key_context = object.__getattribute__(
                        value, "_jinest_key_context"
                    )
            return _SequenceProxy(
                self,
                source,
                parent,
                path,
                key_context=sequence_key_context,
                path_kind=path_kind,
                local_vars=local_vars,
                function_scope=function_scope,
                function_origin_source=function_origin_source,
                function_body_source_path=function_body_source_path,
                context_origin_source=context_origin_source,
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
        sequence_key_context: tuple[Any, Any] | None = None,
        local_vars: Mapping[str, Any] | None = None,
        function_scope: _ContainerProxy | None = None,
        function_origin_source: _Source | None = None,
        function_body_source_path: tuple[Any, ...] | None = None,
        context_origin_source: _Source | None = None,
    ) -> Any:
        if not self._is_container(value):
            return self._copy_scalar(value)

        if local_vars is None:
            local_vars = object.__getattribute__(parent, "_jinest_binding").frame.local_vars
        if function_scope is None:
            function_scope = object.__getattribute__(parent, "_jinest_binding").frame.function_scope
        if function_origin_source is None:
            function_origin_source = object.__getattribute__(parent, "_jinest_binding").frame.function_origin_source
        if function_body_source_path is None:
            function_body_source_path = object.__getattribute__(parent, "_jinest_binding").frame.function_body_source_path
        if context_origin_source is None:
            context_origin_source = object.__getattribute__(parent, "_jinest_binding").frame.context_origin_source

        if isinstance(value, _ContainerProxy):
            source = object.__getattribute__(value, "_jinest_source")
            source_document = source.document
            raw = source.raw
            source_origin = source.resolver
            value_frame = object.__getattribute__(value, "_jinest_binding").frame
            value_local_vars = value_frame.local_vars
            value_body_source_path = value_frame.function_body_source_path
            value_context_origin_source = value_frame.context_origin_source
            if value_context_origin_source is None:
                function_origin = value_frame.function_origin_source
                in_function_body = (
                    function_origin is not None
                    and value_body_source_path is not None
                    and self._same_source_tree(source, function_origin)
                    and source.source_path[: len(value_body_source_path)]
                    == value_body_source_path
                )
                value_context_origin_source = (
                    function_origin if in_function_body else source
                )
            context_origin_source = value_context_origin_source
            if value_body_source_path is not None:
                # A structural function result carries its parameter frame,
                # but its temporary call-site scope must never leak into the
                # destination binding. Merge outer locals only when rebinding
                # from inside another function.
                if value_local_vars is not None:
                    if local_vars is None:
                        local_vars = value_local_vars
                    elif local_vars is not value_local_vars:
                        merged_locals = dict(local_vars)
                        merged_locals.update(value_local_vars)
                        local_vars = merged_locals
                function_scope = None
                function_body_source_path = value_body_source_path
            else:
                # Rebinding a node produced inside a compose/function frame
                # must retain that node's locals.  Merge them over the
                # destination frame so axis/parameter names remain visible
                # and continue to shadow inherited names normally.
                if value_local_vars is not None:
                    if local_vars is None:
                        local_vars = value_local_vars
                    elif local_vars is not value_local_vars:
                        merged_locals = dict(local_vars)
                        merged_locals.update(value_local_vars)
                        local_vars = merged_locals
                if function_scope is None:
                    function_scope = object.__getattribute__(value, "_jinest_binding").frame.function_scope
            if function_origin_source is None:
                function_origin_source = object.__getattribute__(value, "_jinest_binding").frame.function_origin_source
            if function_body_source_path is None:
                function_body_source_path = object.__getattribute__(value, "_jinest_binding").frame.function_body_source_path
            if context_origin_source is None:
                context_origin_source = value_context_origin_source
        else:
            raw = value
            source_origin = origin or self
            source_origin_path = tuple(source_path or ())
            parent_source = object.__getattribute__(parent, "_jinest_source")
            source_document = (
                parent_source.document
                if source_origin is parent_source.resolver
                else None
            )

        if isinstance(value, _ContainerProxy):
            source_origin_path = source.source_path

        children = object.__getattribute__(parent, "_jinest_children")
        cache_token = (
            id(raw),
            id(source_origin),
            source_origin_path,
            sequence_key_context,
            id(source_document) if source_document is not None else None,
            id(local_vars) if local_vars is not None else None,
            id(function_scope) if function_scope is not None else None,
            id(function_origin_source) if function_origin_source is not None else None,
            function_body_source_path,
            id(context_origin_source) if context_origin_source is not None else None,
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
            sequence_key_context=sequence_key_context,
            source_document=source_document,
            path_kind=path_kind,
            local_vars=local_vars,
            function_scope=function_scope,
            function_origin_source=function_origin_source,
            function_body_source_path=function_body_source_path,
            context_origin_source=context_origin_source,
        )
        if function_body_source_path is not None and isinstance(proxy, _ContainerProxy):
            # Nested body fields resolve through the fresh destination
            # parent, never through the temporary proxy created at the call
            # site. The parameter frame remains on the structural node.
            object.__getattribute__(proxy, "_jinest_binding").frame.function_scope = parent
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

    def _clone_mapping_key(self, key: Any, *, active: set[int]) -> Any:
        """Clone one generated mapping key without non-strict data loss."""
        cloned = self._clone_unresolved(key, active=active)
        # ``strict=False`` deliberately turns unsupported *values* into None,
        # but doing that to keys can silently overwrite a different entry.
        if cloned is None and key is not None:
            raise JinestError(
                f"Unsupported mapping key of type {type(key).__name__}"
            )
        try:
            hash(cloned)
        except TypeError as exc:
            raise JinestError(
                f"Unsupported mapping key of type {type(key).__name__}"
            ) from exc
        return cloned

    # Mapping schema and lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_key(key: Any) -> re.Match[str] | None:
        if not isinstance(key, str) or _literal_syntax_key(key):
            return None
        match = _MERGE_RE.fullmatch(key)
        if match is not None:
            return match
        if key.startswith("<<") or key.startswith(".<<"):
            if _INVALID_LEGACY_MERGE_RE.fullmatch(key):
                detail = "place '!' before the numeric order"
            else:
                detail = "use <<, <<!, <<N, or <<!N followed by $, ^, =, or []"
            raise JinestError(f"Invalid merge declaration {key!r}; {detail}")
        return None

    @classmethod
    def _template_key(cls, key: Any) -> tuple[Any, str] | None:
        if (
            not isinstance(key, str)
            or _literal_syntax_key(key)
            or cls._merge_key(key)
        ):
            return None
        for suffix, mode in _FIELD_SUFFIX_MODES:
            if key.endswith(suffix):
                return key[:-1], mode
        return None

    def _mapping_entries(
        self,
        source: _Source,
        bind: _MappingProxy,
        *,
        local_vars: Mapping[str, Any] | None = None,
        context_origin_source: _Source | None = None,
    ) -> tuple[_MappingKeyEntry, ...]:
        """Build a destination-bound index of static, raw, and dynamic keys."""
        indexes = object.__getattribute__(bind, "_jinest_key_indexes")
        cache_key = (
            source.instance_id,
            id(local_vars) if local_vars is not None else None,
            id(context_origin_source)
            if context_origin_source is not None
            else None,
        )
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
                and _field_control_key(entry.source_key) is None
                and _parse_compose_declaration(
                    entry.source_key, source.raw[entry.source_key]
                )
                is None
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
                compose = _parse_compose_declaration(
                    source_key, source.raw[source_key]
                )
                if compose is not None:
                    add(_MappingKeyEntry(source_key, compose.name, compose=True))
                    continue
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
                    local_vars=local_vars,
                    context_origin_source=context_origin_source,
                )
                if not isinstance(key, str):
                    raise JinestError(
                        f"Dynamic mapping key {source_key!r} resolved to "
                        f"{type(key).__name__}, expected a string"
                    )
                add(_MappingKeyEntry(source_key, key, dynamic=True))

            # Do not publish a completed index until every runtime logical-name
            # invariant has passed.
            self._validate_mapping_entry_claims(source, entries)
        except Exception as exc:
            indexes.pop(cache_key, None)
            self._annotate_error(
                exc,
                path=_format_path_segments(
                    "root",
                    object.__getattribute__(bind, "_jinest_path"),
                ),
                file=self._source_label_for(source),
            )
            raise

        entries.sort(key=lambda entry: source_positions[entry.source_key])
        result = tuple(entries)
        indexes[cache_key] = result
        return result

    def _mapping_entry_claim(
        self,
        source: _Source,
        entry: _MappingKeyEntry,
    ) -> tuple[tuple[str, Any], bool, str] | None:
        """Return ``(namespace/name, alternatives_allowed, kind)`` for an entry."""
        source_key = entry.source_key
        value = source.raw[source_key]

        if entry.compose:
            return ("public", entry.key), False, "compose"

        if not entry.raw and not entry.dynamic:
            if self._merge_key(source_key) is not None:
                return None
            if _field_control_key(source_key) is not None:
                return None
            function = _parse_function_declaration(source_key)
            if function is not None:
                return ("public", function.name), False, "function"
            template_info = self._template_key(source_key)
            logical = template_info[0] if template_info else entry.key
            hidden = isinstance(logical, str) and logical.startswith(".")
            if hidden:
                logical = logical[1:]
            namespace = "hidden" if hidden else "public"
        else:
            # Raw and dynamic results are final literal keys.  In particular,
            # a leading dot must not move them into the hidden namespace.
            logical = entry.key
            namespace = "public"
            template_info = None

        concrete = entry.raw or entry.dynamic or template_info is None
        self_spec = _parse_self_declaration(value) if concrete else None
        if self_spec is not None:
            if self_spec.mode == "structural":
                return (namespace, logical), False, "self function"
            if self_spec.mode == "compose_structural":
                return (namespace, logical), False, "self compose"

        alternatives_allowed = not entry.raw and not entry.dynamic
        return (namespace, logical), alternatives_allowed, "field"

    def _validate_mapping_entry_claims(
        self,
        source: _Source,
        entries: Sequence[_MappingKeyEntry],
    ) -> None:
        """Reject runtime key collisions, including destination-bound keys."""
        claimed: dict[
            tuple[str, Any], tuple[_MappingKeyEntry, bool, str]
        ] = {}
        special_kinds = {"function", "compose", "self function", "self compose"}

        def collision_error(
            name: tuple[str, Any],
            previous: tuple[_MappingKeyEntry, bool, str],
            entry: _MappingKeyEntry,
            kind: str,
        ) -> JinestError:
            namespace, logical = name
            display = f".{logical}" if namespace == "hidden" else logical
            previous_entry, _, previous_kind = previous
            collision = (
                "dynamic mapping key"
                if entry.dynamic or previous_entry.dynamic
                else "logical name"
            )
            return JinestError(
                f"Duplicate {collision} {display!r} from "
                f"{previous_entry.source_key!r} ({previous_kind}) and "
                f"{entry.source_key!r} ({kind})"
            )

        for entry in entries:
            claim = self._mapping_entry_claim(source, entry)
            if claim is None:
                continue
            name, alternatives_allowed, kind = claim
            namespace, logical = name

            # A public helper/compose cannot coexist with a hidden field of the
            # same logical name: hidden lookup would make the declaration
            # unreachable. Hidden self declarations may intentionally shadow a
            # public output field, following normal hidden-field semantics.
            cross_name = (
                ("hidden", logical) if namespace == "public" else ("public", logical)
            )
            cross = claimed.get(cross_name)
            if (
                cross is not None
                and (
                    (namespace == "public" and kind in special_kinds)
                    or (namespace == "hidden" and cross[2] in special_kinds)
                )
            ):
                raise collision_error(name, cross, entry, kind)

            previous = claimed.get(name)
            if previous is None:
                claimed[name] = (entry, alternatives_allowed, kind)
                continue
            if alternatives_allowed and previous[1]:
                # Ordinary name/name^/name$/name@ alternatives intentionally
                # coexist and are handled by local declaration priority.
                continue
            raise collision_error(name, previous, entry, kind)

    def _apply_render_layer(
        self,
        scope: _ContainerProxy,
        value: Any,
        *,
        mode: str,
        origin_source: _Source,
        source_key: Any,
        context_path: tuple[Any, ...] | None,
        keyname: Any | None,
        effective_key: Any | None,
        keymode: str | None,
        local_vars: Mapping[str, Any] | None = None,
        context_origin_source: _Source | None = None,
    ) -> Any:
        """Apply an outer mode to a result produced by an inner layer."""
        if not _valid_evaluator_body(value, mode):
            marker = _FUNCTION_MODE_MARKERS[mode]
            path = context_path or object.__getattribute__(scope, "_jinest_path")
            raise JinestTemplateError(
                f"Nested {marker} layer for {source_key!r} requires "
                f"{_evaluator_body_requirement(mode)}, "
                f"got {type(value).__name__}",
                path=_format_path_segments("global_root", path),
                file=self._source_label_for(origin_source),
            )
        return self._evaluators.apply(
            self,
            EvaluationPlan(EvaluatorKind(mode), value, source_key),
            scope=scope,
            origin_source=origin_source,
            source_path=origin_source.source_path + (source_key,),
            context_path=context_path,
            keyname=keyname,
            effective_key=effective_key,
            keymode=keymode,
            local_vars=local_vars,
            context_origin_source=context_origin_source,
        )

    def _array_transform_list(
        self,
        value: Any,
        *,
        mode: str,
        source_key: Any,
        elements: bool = False,
    ) -> list[Any]:
        """Read a strict list input without materializing nested containers."""
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, _SequenceProxy):
            raw = object.__getattribute__(
                value, "_jinest_source"
            ).raw
            if isinstance(raw, (list, tuple)):
                return [value[index] for index in range(len(value))]
        expected = "list or tuple elements" if elements else "a list or tuple"
        marker = _ARRAY_TRANSFORM_MARKERS[mode]
        raise JinestError(
            f"Array suffix {marker!r} for {source_key!r} requires {expected}, "
            f"got {type(value).__name__}"
        )

    def _apply_array_transform(
        self,
        value: Any,
        *,
        mode: str,
        source_key: Any,
    ) -> Any:
        """Apply one strict array combinator while preserving nested nodes."""
        outer = self._array_transform_list(
            value, mode=mode, source_key=source_key
        )
        marker = _ARRAY_TRANSFORM_MARKERS[mode]

        if mode == "join":
            parts: list[str] = []
            for index, item in enumerate(outer):
                if not isinstance(item, str):
                    raise JinestError(
                        f"Array suffix {marker!r} for {source_key!r} requires "
                        f"string elements, got {type(item).__name__} at index {index}"
                    )
                parts.append(item)
            return "".join(parts)

        axes: list[list[Any]] = []
        for index, item in enumerate(outer):
            try:
                axis = self._array_transform_list(
                    item, mode=mode, source_key=source_key, elements=True
                )
            except JinestError as exc:
                raise JinestError(
                    f"Array suffix {marker!r} for {source_key!r} requires "
                    f"list or tuple elements, got {type(item).__name__} at index {index}"
                ) from exc
            axes.append(axis)

        if mode == "flatten":
            flattened: list[Any] = []
            for axis in axes:
                flattened.extend(axis)
            return flattened

        if mode == "product":
            return [list(items) for items in product(*axes)]

        if mode == "zip":
            if not axes:
                return []
            length = len(axes[0])
            for index, axis in enumerate(axes[1:], start=1):
                if len(axis) != length:
                    raise JinestError(
                        f"Array suffix {marker!r} for {source_key!r} requires "
                        f"equal axis lengths, got {length} and {len(axis)} "
                        f"at axis {index}"
                    )
            return [list(items) for items in zip(*axes)]

        raise JinestError(f"Unsupported array transform mode {mode!r}")

    def _resolve_layer_input(
        self,
        scope: _ContainerProxy,
        value: Any,
        *,
        origin_source: _Source,
        source_key: Any,
        context_path: tuple[Any, ...] | None = None,
        keyname: Any | None = None,
        effective_key: Any | None = None,
        local_vars: Mapping[str, Any] | None = None,
        context_origin_source: _Source | None = None,
    ) -> LayerResult:
        """Resolve an inner inline/self layer.

        A typed result lets a field/array suffix act as a true outer layer
        while preserving escaped inline literals.
        """
        escaped = _escaped_inline_literal(value)
        if escaped is not None:
            return LayerResult(escaped, escaped_literal=True)

        self_spec = _parse_self_declaration(value)
        if self_spec is not None:
            if self_spec.mode in {"native", "text", "script"}:
                inner = self._resolve_layer_input(
                    scope,
                    self_spec.payload,
                    origin_source=origin_source,
                    source_key=source_key,
                    context_path=context_path,
                    keyname=keyname,
                    effective_key=effective_key,
                    local_vars=local_vars,
                    context_origin_source=context_origin_source,
                )
                mode_marker = {"native": "$", "text": "@", "script": "^"}[self_spec.mode]
                result = self._apply_render_layer(
                    scope,
                    inner.value,
                    mode=self_spec.mode,
                    origin_source=origin_source,
                    source_key=self_spec.source_key,
                    context_path=context_path,
                    keyname=keyname,
                    effective_key=effective_key,
                    keymode=mode_marker,
                    local_vars=local_vars,
                    context_origin_source=context_origin_source,
                )
                return LayerResult(result, applied=True)
            if self_spec.mode == "structural":
                function = self_spec.payload
                if not isinstance(function, _FunctionSpec):
                    raise JinestError(f"Malformed self structural function {self_spec.source_key!r}")
                spec = _FunctionSpec(
                    name=str(keyname),
                    source_key=self_spec.source_key,
                    template=function.template,
                    mode="structural",
                    parameters=function.parameters,
                )
                return LayerResult(self._function_value(origin_source, spec), applied=True)
            if self_spec.mode == "compose_structural":
                compose = self_spec.payload
                if not isinstance(compose, _ComposeSpec):
                    raise JinestError(f"Malformed self structural compose {self_spec.source_key!r}")
                spec = _ComposeSpec(
                    name=str(keyname),
                    source_key=self_spec.source_key,
                    template=compose.template,
                    mode="structural",
                    axes=compose.axes,
                )
                evaluation_scope = scope
                while isinstance(evaluation_scope, _SequenceProxy):
                    evaluation_scope = object.__getattribute__(
                        evaluation_scope, "_jinest_parent"
                    )
                if not isinstance(evaluation_scope, _MappingProxy):
                    raise JinestError(
                        f"Self structural compose {self_spec.source_key!r} "
                        "requires a mapping evaluation scope"
                    )
                destination_key = (
                    source_key if isinstance(scope, _SequenceProxy) else keyname
                )
                return LayerResult(self._resolve_compose(
                    spec,
                    origin_source,
                    evaluation_scope,
                    keyname,
                    destination_parent=scope,
                    destination_key=destination_key,
                    context_origin_source=context_origin_source,
                ), applied=True)
            raise JinestError(f"Unsupported self declaration mode {self_spec.mode!r}")

        directive = _inline_directive(value)
        if directive is None:
            return LayerResult(value)
        mode, template = directive
        result = self._evaluators.apply(
            self,
            EvaluationPlan(EvaluatorKind(mode), template, source_key),
            scope=scope,
            origin_source=origin_source,
            source_path=origin_source.source_path + (source_key,),
            context_path=context_path,
            keyname=keyname,
            effective_key=effective_key,
            keymode=_FUNCTION_MODE_MARKERS[mode],
            local_vars=local_vars,
            context_origin_source=context_origin_source,
        )
        return LayerResult(result, applied=True)

    def _record_schema_messages(self, source: _Source) -> None:
        """Report declarations that are present but cannot become effective."""
        raw = source.raw
        if not isinstance(raw, Mapping):
            return
        declarations: dict[tuple[str, bool], list[tuple[int, str]]] = {}
        priority = {"": 0}
        priority.update(
            {suffix: index for index, (suffix, _) in enumerate(_FIELD_SUFFIX_MODES, 1)}
        )
        priority_text = "name > name^ > name$ > name@ > name* > name+ > name% > name~"
        for key in raw:
            if (
                not isinstance(key, str)
                or _literal_syntax_key(key)
                or self._merge_key(key)
                or _field_control_key(key) is not None
            ):
                continue
            if (
                _parse_compose_declaration(key, raw[key]) is not None
                or _parse_function_declaration(key) is not None
            ):
                continue
            mode = ""
            base = key
            for suffix, _ in _FIELD_SUFFIX_MODES:
                if key.endswith(suffix):
                    mode = suffix
                    base = key[:-1]
                    break
            if mode == "":
                self_spec = _parse_self_declaration(raw[key])
                if self_spec is not None and self_spec.mode in {
                    "structural",
                    "compose_structural",
                }:
                    # These are declarations, not ordinary concrete fields, so
                    # field-mode priority does not apply to them. Suffixed
                    # alternatives are deliberately not parsed here because a
                    # higher-priority field may suppress them.
                    continue
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
                    f"{priority_text}",
                    dedupe_key=("warning", source.document_id, winner, suppressed),
                    source=source,
                )

        public_names = {logical for (logical, hidden) in declarations if not hidden}
        hidden_names = {logical for (logical, hidden) in declarations if hidden}
        for logical in sorted(public_names & hidden_names):
            self._record_message(
                "hint",
                f"Hidden field '.{logical}' takes priority over field {logical!r} "
                "in template calculations; the public field remains independently materialized",
                dedupe_key=("hint", source.document_id, logical),
                source=source,
            )

    def _schema(self, raw: Mapping[Any, Any]) -> CompiledMapping:
        return self._syntax.compile(raw)

    def _schema_for_source(self, source: _Source) -> CompiledMapping:
        try:
            schema = self._schema(source.raw)
            self._record_schema_messages(source)
            return schema
        except Exception as exc:
            path, file = self._source_location(source)
            self._annotate_error(exc, path=path, file=file)
            raise

    @staticmethod
    def _ordered_layers(layers: Sequence[_LayerSpec]) -> tuple[_LayerSpec, ...]:
        """Order one layer family using the established reverse-lookup rules."""
        return tuple(
            sorted(
                layers,
                key=lambda item: (
                    item.order,
                    not item.multiple,
                    item.position,
                    -1 if item.item_index is None else item.item_index,
                ),
            )
        )

    def _layer_stack(
        self,
        source: _Source,
        bind: _MappingProxy,
        *,
        local_vars: Mapping[str, Any] | None = None,
        context_origin_source: _Source | None = None,
    ) -> tuple[tuple[_LayerSpec, ...], tuple[_LayerSpec, ...]]:
        """Return the destination-bound, flat merge topology for ``source``.

        A ``<<[]`` declaration contributes one lazy normal layer spec per list
        item.  The topology is available while it is still being built, so a
        later source can use fields inherited from an earlier array layer.
        """
        schema = self._schema_for_source(source)
        cache = object.__getattribute__(bind, "_jinest_binding").cache.normalized_layers
        cache_key = (
            source.instance_id,
            id(local_vars) if local_vars is not None else None,
            id(context_origin_source)
            if context_origin_source is not None
            else None,
        )
        stack = cache.get(cache_key)
        if stack is not None:
            return (
                self._ordered_layers(stack.defaults),
                self._ordered_layers(stack.overrides),
            )

        stack = _LayerStack(
            defaults=[layer for layer in schema.defaults if not layer.multiple],
            overrides=[layer for layer in schema.overrides if not layer.multiple],
        )
        # Publish ordinary layers before resolving a list-producing expression:
        # this mirrors normal lazy merge lookup during re-entrant evaluation.
        cache[cache_key] = stack
        try:
            self._normalize_layer_specs(
                bind,
                source,
                schema.defaults,
                stack.defaults,
                local_vars=local_vars,
                context_origin_source=context_origin_source,
            )
            self._normalize_layer_specs(
                bind,
                source,
                schema.overrides,
                stack.overrides,
                local_vars=local_vars,
                context_origin_source=context_origin_source,
            )
        except Exception:
            cache.pop(cache_key, None)
            raise
        return self._ordered_layers(stack.defaults), self._ordered_layers(stack.overrides)

    def _normalize_layer_specs(
        self,
        bind: _MappingProxy,
        source: _Source,
        specs: Sequence[_LayerSpec],
        destination: list[_LayerSpec],
        *,
        local_vars: Mapping[str, Any] | None = None,
        context_origin_source: _Source | None = None,
    ) -> None:
        """Append flat lazy item specs without evaluating their mappings."""
        for spec in specs:
            if not spec.multiple:
                continue

            declaration_path = object.__getattribute__(bind, "_jinest_path") + (
                spec.source_key,
            )
            layer_result = self._resolve_layer_input(
                bind,
                spec.template,
                origin_source=source,
                source_key=spec.source_key,
                context_path=declaration_path,
                local_vars=local_vars,
                context_origin_source=context_origin_source,
            )
            value = layer_result.value
            sequence: _SequenceProxy
            if isinstance(value, _SequenceProxy):
                raw = object.__getattribute__(value, "_jinest_source").raw
                if not isinstance(raw, list):
                    self._raise_merge_type_error(
                        bind, spec.source_key, value, "a list", source=source
                    )
                sequence = value
            elif isinstance(value, list):
                sequence_value = self._bind_child(
                    bind,
                    spec.source_key,
                    value,
                    origin=source.resolver,
                    source_path=source.source_path + (spec.source_key,),
                    local_vars=local_vars,
                )
                if not isinstance(sequence_value, _SequenceProxy):  # defensive
                    self._raise_merge_type_error(
                        bind, spec.source_key, value, "a list", source=source
                    )
                sequence = sequence_value
            else:
                self._raise_merge_type_error(
                    bind, spec.source_key, value, "a list", source=source
                )

            destination.extend(
                _LayerSpec(
                    source_key=(spec.source_key, index),
                    template=None,
                    order=spec.order,
                    position=spec.position,
                    override=spec.override,
                    mode=EvaluatorKind.NATIVE,
                    hidden=spec.hidden,
                    multiple=True,
                    item_sequence=sequence,
                    item_index=index,
                )
                for index in range(len(sequence))
            )

    def _raise_merge_type_error(
        self,
        bind: _MappingProxy,
        declaration: Any,
        value: Any,
        expected: str,
        *,
        source: _Source | None = None,
    ) -> NoReturn:
        path = object.__getattribute__(bind, "_jinest_path")
        if isinstance(declaration, tuple):
            path += declaration
        else:
            path += (declaration,)
        error_source = source or object.__getattribute__(bind, "_jinest_source")
        raise JinestMergeError(
            f"Merge {_format_path_segments('global_root', path)} produced "
            f"{type(value).__name__}, expected {expected}",
            path=_format_path_segments("root", path),
            file=self._source_label_for(error_source),
        )

    def _compile_mapping(self, raw: Mapping[Any, Any]) -> CompiledMapping:

        defaults: list[_LayerSpec] = []
        overrides: list[_LayerSpec] = []
        functions: list[_FunctionSpec] = []
        composes: list[_ComposeSpec] = []
        function_names: set[str] = set()
        compose_names: set[str] = set()
        public_controls: dict[str, str] = {}
        for position, (key, template) in enumerate(raw.items()):
            compose = _parse_compose_declaration(key, template)
            if compose is not None:
                if compose.name in compose_names:
                    raise JinestError(
                        f"Duplicate compose declaration {compose.name!r} "
                        f"at {_format_path_segments('root', (compose.source_key,))}"
                    )
                compose_names.add(compose.name)
                composes.append(compose)
                continue
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

            control = _field_control_key(key)
            if control is not None:
                logical, channel, behavior = control
                if template is not None:
                    raise JinestError(
                        f"Field {key!r} is a {behavior} control and requires a null body"
                    )
                if channel == "public":
                    previous = public_controls.get(logical)
                    if previous is not None and previous != behavior:
                        raise JinestError(
                            f"Conflicting declarations for public field {logical!r}: "
                            "DELETE and HIDE"
                        )
                    public_controls[logical] = behavior

            match = self._merge_key(key)
            if match is None:
                continue
            order_text = match.group("order")
            order = int(order_text) if order_text else 0
            mode_text = match.group("mode")
            multiple = mode_text == "[]"
            direct = mode_text == "="
            if direct and not isinstance(template, Mapping):
                raise JinestMergeError(
                    f"Direct merge {key!r} requires a mapping body, got "
                    f"{type(template).__name__}"
                )
            spec = _LayerSpec(
                source_key=key,
                template=template,
                order=order,
                position=position,
                override=bool(match.group("leading_override")),
                mode=(
                    EvaluatorKind.SCRIPT
                    if mode_text == "^"
                    else EvaluatorKind.NATIVE if not direct else None
                ),
                direct=direct,
                hidden=bool(match.group("hidden")),
                multiple=multiple,
            )
            (overrides if spec.override else defaults).append(spec)

        # A function and an ordinary declaration cannot share one logical name.
        ordinary_names: dict[str, Any] = {}
        for key in raw:
            if (
                _literal_syntax_key(key)
                or _parse_compose_declaration(key, raw[key]) is not None
                or _parse_function_declaration(key) is not None
            ):
                continue
            if self._merge_key(key):
                continue
            if _field_control_key(key) is not None:
                continue
            template_info = self._template_key(key)
            logical = template_info[0] if template_info else key
            if isinstance(logical, str) and logical.startswith("."):
                logical = logical[1:]
            if isinstance(logical, str):
                ordinary_names.setdefault(logical, key)
        for compose in composes:
            conflict = ordinary_names.get(compose.name)
            if conflict is not None:
                raise JinestError(
                    f"Compose {compose.name!r} at {compose.source_key!r} "
                    f"conflicts with field declaration {conflict!r}"
                )
            ordinary_names[compose.name] = compose.source_key
        for function in functions:
            conflict = ordinary_names.get(function.name)
            if conflict is not None:
                raise JinestError(
                    f"Function {function.name!r} at {function.source_key!r} "
                    f"conflicts with field declaration {conflict!r}"
                )

        # Lookup walks each family in reverse.  Place an array source before a
        # single source at the same numeric order so the single declaration
        # has the documented effective priority; array items retain list order.
        defaults.sort(key=lambda item: (item.order, not item.multiple, item.position))
        overrides.sort(key=lambda item: (item.order, not item.multiple, item.position))
        schema = CompiledMapping(
            raw, tuple(defaults), tuple(overrides), tuple(functions), tuple(composes)
        )
        return schema

    def _local_candidates(
        self,
        source: _Source,
        bind: _MappingProxy,
        key: Any,
        *,
        channel: str,
        source_hidden: bool,
        local_vars: Mapping[str, Any] | None = None,
        context_origin_source: _Source | None = None,
    ) -> tuple[_Candidate, ...]:
        """Return value/control candidates for one independent field channel.

        ``.x`` and ``x`` are distinct channels.  A hidden layer redirects
        ordinary values from its source into the hidden channel, but never
        redirects DELETE/HIDE controls: those operate on public output only.
        """
        entries = self._mapping_entries(
            source,
            bind,
            local_vars=local_vars,
            context_origin_source=context_origin_source,
        )

        def value_candidate(
            physical: Any,
            *,
            explicit_hidden: bool,
        ) -> _Candidate | None:
            for entry in entries:
                if entry.key != physical:
                    continue
                if entry.raw or entry.dynamic:
                    # A raw/dynamic leading dot remains a literal final key,
                    # not an explicit hidden declaration.  A hidden *layer*
                    # may still redirect a literal public key into hidden.
                    if not explicit_hidden:
                        return _Candidate(
                            entry.source_key,
                            source.raw[entry.source_key],
                            "concrete",
                        )
                    continue
                if (
                    self._merge_key(entry.source_key) is not None
                    or _field_control_key(entry.source_key) is not None
                    or _parse_compose_declaration(
                        entry.source_key, source.raw[entry.source_key]
                    ) is not None
                    or _parse_function_declaration(entry.source_key) is not None
                ):
                    continue
                if self._template_key(entry.source_key) is None:
                    return _Candidate(
                        entry.source_key,
                        source.raw[entry.source_key],
                        "concrete",
                    )

            if isinstance(physical, str):
                for suffix, mode in _FIELD_SUFFIX_MODES:
                    expected = f"{physical}{suffix}"
                    for entry in entries:
                        if (
                            entry.key == expected
                            and not entry.raw
                            and not entry.dynamic
                        ):
                            return _Candidate(
                                entry.source_key,
                                source.raw[entry.source_key],
                                mode,
                            )
            return None

        candidates: list[_Candidate] = []
        if not isinstance(key, str) or key.startswith("."):
            # Explicit physical access remains compatible with the historical
            # mapping protocol. It never activates hidden-channel fallback.
            candidate = value_candidate(key, explicit_hidden=False)
            return () if candidate is None else (candidate,)

        if channel not in {"hidden", "public"}:  # defensive invariant
            raise JinestError(f"Unsupported field channel {channel!r}")

        # Controls precede values in their own channel. A tombstone stops
        # lower candidates; a public HIDE preserves lookup but masks output.
        for entry in entries:
            if entry.raw or entry.dynamic:
                continue
            control = _field_control_key(entry.source_key)
            if (
                control is None
                or control[0] != key
                or control[1] != channel
            ):
                continue
            candidates.append(
                _Candidate(
                    entry.source_key,
                    source.raw[entry.source_key],
                    "concrete",
                    behavior=control[2],
                )
            )

        if channel == "hidden":
            explicit = value_candidate(f".{key}", explicit_hidden=True)
            if explicit is not None:
                candidates.append(explicit)
            if source_hidden:
                inherited_public = value_candidate(key, explicit_hidden=False)
                if inherited_public is not None:
                    candidates.append(inherited_public)
            return tuple(candidates)

        if not source_hidden:
            for compose in self._schema_for_source(source).composes:
                if compose.name == key:
                    candidates.append(
                        _Candidate(
                            compose.source_key,
                            compose.template,
                            f"compose_{compose.mode}",
                        )
                    )
                    break
            public_value = value_candidate(key, explicit_hidden=False)
            if public_value is not None:
                candidates.append(public_value)
        return tuple(candidates)

    def _local_function(self, source: _Source, key: Any) -> _FunctionSpec | None:
        if not isinstance(key, str) or key.startswith("."):
            return None
        schema = self._schema_for_source(source)
        for function in schema.functions:
            if function.name == key:
                return function
        return None

    def _function_value(
        self,
        source: _Source,
        spec: _FunctionSpec,
        scope: _ContainerProxy | None = None,
    ) -> JinestFunction:
        return JinestFunction(self, spec, source, scope)

    def _iter_field_candidates(
        self,
        source: _Source,
        key: Any,
        *,
        bind: _MappingProxy,
        active: set[tuple[Any, ...]],
        channel: str,
        source_hidden: bool = False,
        local_vars: Mapping[str, Any] | None = None,
        context_origin_source: _Source | None = None,
    ) -> Iterator[_CandidateLocation]:
        """Yield candidates in one channel from highest to lowest precedence."""
        if context_origin_source is None:
            context_origin_source = object.__getattribute__(
                bind, "_jinest_binding"
            ).frame.context_origin_source
        token = (
            source.instance_id,
            self._hashable_key(key),
            channel,
            source_hidden,
            id(local_vars) if local_vars is not None else None,
            (
                context_origin_source.instance_id
                if context_origin_source is not None
                else None
            ),
        )
        if token in active:
            return
        active.add(token)
        try:
            defaults, overrides = self._layer_stack(
                source,
                bind,
                local_vars=local_vars,
                context_origin_source=context_origin_source,
            )
            for layer in reversed(overrides):
                layer_value = self._evaluate_layer(
                    bind,
                    source,
                    layer,
                    local_vars=local_vars,
                    context_origin_source=context_origin_source,
                )
                yield from self._iter_field_candidates(
                    layer_value.source,
                    key,
                    bind=bind,
                    active=active,
                    channel=channel,
                    source_hidden=source_hidden or layer_value.hidden,
                    local_vars=layer_value.local_vars,
                    context_origin_source=(
                        layer_value.context_origin_source or context_origin_source
                    ),
                )

            # Functions are helpers, never materialized fields. They remain
            # reachable through public lookup regardless of layer visibility.
            if channel == "public":
                function = self._local_function(source, key)
                if function is not None:
                    yield _CandidateLocation(
                        _Candidate(function.source_key, function, "function"),
                        source,
                        local_vars,
                        context_origin_source,
                    )

            for candidate in self._local_candidates(
                source,
                bind,
                key,
                channel=channel,
                source_hidden=source_hidden,
                local_vars=local_vars,
                context_origin_source=context_origin_source,
            ):
                yield _CandidateLocation(
                    candidate,
                    source,
                    local_vars,
                    context_origin_source,
                )

            for layer in reversed(defaults):
                layer_value = self._evaluate_layer(
                    bind,
                    source,
                    layer,
                    local_vars=local_vars,
                    context_origin_source=context_origin_source,
                )
                yield from self._iter_field_candidates(
                    layer_value.source,
                    key,
                    bind=bind,
                    active=active,
                    channel=channel,
                    source_hidden=source_hidden or layer_value.hidden,
                    local_vars=layer_value.local_vars,
                    context_origin_source=(
                        layer_value.context_origin_source or context_origin_source
                    ),
                )
        finally:
            active.remove(token)

    def _find_field(
        self,
        source: _Source,
        key: Any,
        *,
        bind: _MappingProxy,
        channel: str,
        local_vars: Mapping[str, Any] | None = None,
        context_origin_source: _Source | None = None,
    ) -> _FieldMatch | object:
        """Select one candidate in the hidden or public field channel."""
        masked = False
        for location in self._iter_field_candidates(
            source,
            key,
            bind=bind,
            active=set(),
            channel=channel,
            local_vars=local_vars,
            context_origin_source=context_origin_source,
        ):
            behavior = location.candidate.behavior
            if channel == "public" and behavior == "hide":
                masked = True
                continue
            if behavior == "delete":
                return _MISSING
            return _FieldMatch(location, masked)
        return _MISSING

    def _scope_has_logical(self, scope: _MappingProxy, key: Any) -> bool:
        resolved = object.__getattribute__(scope, "_jinest_resolved")
        if key in resolved:
            return True
        source = object.__getattribute__(scope, "_jinest_source")
        context_origin_source = object.__getattribute__(
            scope, "_jinest_binding"
        ).frame.context_origin_source
        return self._find_field(
            source,
            key,
            bind=scope,
            channel="hidden",
            context_origin_source=context_origin_source,
        ) is not _MISSING or self._find_field(
            source,
            key,
            bind=scope,
            channel="public",
            context_origin_source=context_origin_source,
        ) is not _MISSING

    def _get_field(self, scope: _MappingProxy, key: Any) -> Any:
        return self._get_cached_field(scope, key, public=False)

    def _get_public_field(self, scope: _MappingProxy, key: Any) -> Any:
        """Resolve the public-channel candidate selected by ``_public_keys``."""
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
            value = self._lookup(source, key, bind=scope, public=public)
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
        public: bool = False,
        local_vars: Mapping[str, Any] | None = None,
        context_origin_source: _Source | None = None,
    ) -> Any:
        """Resolve normal lookup (hidden then public) or public-only lookup."""
        channels = ("public",) if public else ("hidden", "public")
        for channel in channels:
            match = self._find_field(
                source,
                key,
                bind=bind,
                channel=channel,
                local_vars=local_vars,
                context_origin_source=context_origin_source,
            )
            if match is _MISSING:
                continue
            if not isinstance(match, _FieldMatch):  # defensive invariant
                raise JinestError("Malformed field lookup result")
            location = match.location
            return self._resolve_candidate(
                location.candidate,
                location.source,
                bind,
                key,
                local_vars=location.local_vars,
                context_origin_source=location.context_origin_source,
            )
        return _MISSING

    def _resolve_compose(
        self,
        spec: _ComposeSpec,
        source: _Source,
        bind: _MappingProxy,
        logical_key: Any,
        *,
        destination_parent: _ContainerProxy | None = None,
        destination_key: Any = _MISSING,
        context_origin_source: _Source | None = None,
    ) -> Any:
        """Expand one compose declaration through ordinary lazy bindings."""
        declaration_path = source.source_path + (spec.source_key,)
        axis_values: list[list[Any]] = []
        for axis in spec.axes:
            value = self._render(
                bind,
                axis.source,
                mode="native",
                origin_source=source,
                source_path=declaration_path + (f"axis:{axis.name}",),
                keyname=logical_key,
                effective_key=spec.source_key,
                keymode="=" if spec.mode == "structural" else "@",
                prepare_native=False,
                context_origin_source=context_origin_source,
            )
            try:
                axis_values.append(list(value))
            except TypeError as exc:
                raise JinestError(
                    f"Compose axis {axis.name!r} in {spec.source_key!r} "
                    f"must resolve to an iterable, got {type(value).__name__}"
                ) from exc

        inherited_locals = object.__getattribute__(bind, "_jinest_binding").frame.local_vars
        if destination_parent is None:
            destination_parent = bind
        if destination_key is _MISSING:
            destination_key = logical_key
        destination_path = object.__getattribute__(
            destination_parent, "_jinest_path"
        ) + (destination_key,)
        path_kind = object.__getattribute__(destination_parent, "_jinest_path_kind")

        combination_length = 1
        for values in axis_values:
            combination_length *= len(values)

        if spec.mode == "text":
            parts: list[str] = []
            for combination_index0, indexes in enumerate(
                product(*(range(len(values)) for values in axis_values))
            ):
                frame = _compose_local_frame(
                    inherited_locals,
                    spec,
                    axis_values,
                    indexes,
                    combination_index0,
                    combination_length,
                )
                parts.append(
                    self._render(
                        bind,
                        spec.template,
                        mode="text",
                        origin_source=source,
                        source_path=declaration_path,
                        keyname=logical_key,
                        effective_key=spec.source_key,
                        keymode="@",
                        local_vars=frame,
                        context_origin_source=context_origin_source,
                    )
                )
            return "".join(parts)

        if isinstance(spec.template, Mapping):
            combined: dict[Any, Any] = {}
            seen: set[Any] = set()
            structural_kind = "mapping"
        else:
            combined = []
            seen = set()
            structural_kind = "list"

        for combination_index0, indexes in enumerate(
            product(*(range(len(values)) for values in axis_values))
        ):
            frame = _compose_local_frame(
                inherited_locals,
                spec,
                axis_values,
                indexes,
                combination_index0,
                combination_length,
            )
            body = self._wrap(
                spec.template,
                parent=destination_parent,
                path=destination_path,
                origin=source.resolver,
                source_path=declaration_path,
                source_document=source.document,
                path_kind=path_kind,
                local_vars=frame,
                function_scope=bind,
                function_origin_source=source,
                context_origin_source=context_origin_source,
            )
            if structural_kind == "list":
                if not isinstance(body, _SequenceProxy):
                    raise JinestError(
                        f"Structural compose {spec.source_key!r} produced a non-list body"
                    )
                combined.extend(body[index] for index in range(len(body)))
                continue

            if not isinstance(body, _MappingProxy):
                raise JinestError(
                    f"Structural compose {spec.source_key!r} produced a non-mapping body"
                )
            for key in self._public_keys(
                body, context_origin_source=context_origin_source
            ):
                if key in seen:
                    raise JinestError(
                        f"Duplicate dynamic mapping key {key!r} from compose "
                        f"{spec.source_key!r}"
                    )
                seen.add(key)
                combined[key] = self._get_public_field(body, key)

        return self._bind_child(
            destination_parent,
            destination_key,
            combined,
            origin=source.resolver,
            source_path=declaration_path,
            context_origin_source=context_origin_source,
        )

    def _resolve_candidate(
        self,
        candidate: _Candidate,
        source: _Source,
        bind: _MappingProxy,
        logical_key: Any,
        *,
        local_vars: Mapping[str, Any] | None = None,
        context_origin_source: _Source | None = None,
    ) -> Any:
        if candidate.mode == "function":
            if not isinstance(candidate.template, _FunctionSpec):
                raise JinestError(f"Malformed function declaration {candidate.source_key!r}")
            return self._function_value(source, candidate.template, bind)

        if candidate.mode.startswith("compose_"):
            spec = _parse_compose_declaration(candidate.source_key, candidate.template)
            if spec is None:
                raise JinestError(f"Malformed compose declaration {candidate.source_key!r}")
            return self._resolve_compose(
                spec,
                source,
                bind,
                logical_key,
                context_origin_source=context_origin_source,
            )

        candidate_source_path = source.source_path + (candidate.source_key,)
        layer_result = self._resolve_layer_input(
            bind,
            candidate.template,
            origin_source=source,
            source_key=candidate.source_key,
            context_path=object.__getattribute__(bind, "_jinest_path") + (logical_key,),
            keyname=logical_key,
            effective_key=candidate.source_key,
            local_vars=local_vars,
            context_origin_source=context_origin_source,
        )

        if candidate.mode in _ARRAY_TRANSFORM_MODES.values():
            transform_input = (
                layer_result.value if layer_result.applied else candidate.template
            )
            value = self._apply_array_transform(
                transform_input,
                mode=candidate.mode,
                source_key=candidate.source_key,
            )
            if self._is_container(value):
                return self._bind_child(
                    bind,
                    logical_key,
                    value,
                    origin=source.resolver,
                    source_path=candidate_source_path,
                    local_vars=local_vars,
                    context_origin_source=context_origin_source,
                )
            return value

        if candidate.mode == "concrete":
            if layer_result.applied or layer_result.escaped_literal:
                if self._is_container(layer_result.value):
                    return self._bind_child(
                        bind,
                        logical_key,
                        layer_result.value,
                        origin=source.resolver,
                        source_path=candidate_source_path,
                        sequence_key_context=(logical_key, candidate.source_key),
                        local_vars=local_vars,
                        context_origin_source=context_origin_source,
                    )
                return layer_result.value
            return self._bind_child(
                bind,
                logical_key,
                candidate.template,
                origin=source.resolver,
                source_path=candidate_source_path,
                sequence_key_context=(logical_key, candidate.source_key),
                local_vars=local_vars,
                context_origin_source=context_origin_source,
            )

        mode_marker = {"native": "$", "text": "@", "script": "^"}[candidate.mode]
        if layer_result.applied:
            value = self._apply_render_layer(
                bind,
                layer_result.value,
                mode=candidate.mode,
                origin_source=source,
                source_key=candidate.source_key,
                context_path=object.__getattribute__(bind, "_jinest_path") + (logical_key,),
                keyname=logical_key,
                effective_key=candidate.source_key,
                keymode=mode_marker,
                local_vars=local_vars,
                context_origin_source=context_origin_source,
            )
        elif layer_result.escaped_literal:
            return layer_result.value
        else:
            value = self._render(
                bind,
                candidate.template,
                mode=candidate.mode,
                origin_source=source,
                source_key=candidate.source_key,
                keyname=logical_key,
                effective_key=candidate.source_key,
                keymode=mode_marker,
                local_vars=local_vars,
                context_origin_source=context_origin_source,
            )

        if candidate.mode in {"native", "script"} or self._is_container(value):
            return self._bind_child(
                bind,
                logical_key,
                value,
                origin=source.resolver,
                source_path=candidate_source_path,
                sequence_key_context=(logical_key, candidate.source_key),
                local_vars=local_vars,
                context_origin_source=context_origin_source,
            )
        return value

    def _evaluate_layer(
        self,
        bind: _MappingProxy,
        owner_source: _Source,
        layer: _LayerSpec,
        *,
        local_vars: Mapping[str, Any] | None = None,
        context_origin_source: _Source | None = None,
    ) -> _LayerValue:
        if context_origin_source is None:
            context_origin_source = object.__getattribute__(
                bind, "_jinest_binding"
            ).frame.context_origin_source
        cache = object.__getattribute__(bind, "_jinest_layer_cache")
        cache_key = (
            _DeclarationId(owner_source.document_id, layer.source_key),
            id(local_vars) if local_vars is not None else None,
            id(context_origin_source)
            if context_origin_source is not None
            else None,
        )
        if cache_key in cache:
            cached = cache[cache_key]
            # A recursively requested layer is temporarily empty, mirroring
            # field cycles resolving to None.
            if cached is None:
                return _LayerValue(
                    _Source(
                        owner_source.resolver,
                        _EMPTY_MAPPING,
                        owner_source.source_path + (layer.source_key,),
                        owner_source.document,
                    )
                )
            return cached

        cache[cache_key] = None
        try:
            if layer.item_sequence is not None and layer.item_index is not None:
                value = layer.item_sequence[layer.item_index]
            elif layer.direct:
                value = layer.template
            else:
                if layer.mode is None:  # defensive parser invariant
                    raise JinestError(f"Merge {layer.source_key!r} has no evaluator")
                value = self._render(
                    bind,
                    layer.template,
                    mode=layer.mode,
                    origin_source=owner_source,
                    source_key=layer.source_key,
                    local_vars=local_vars,
                    context_origin_source=context_origin_source,
                )

            if value is None:
                result = _LayerValue(
                    _Source(
                        owner_source.resolver,
                        _EMPTY_MAPPING,
                        owner_source.source_path + (layer.source_key,),
                        owner_source.document,
                    ),
                    hidden=layer.hidden,
                )
            elif isinstance(value, _MappingProxy):
                frame = object.__getattribute__(value, "_jinest_binding").frame
                value_source = object.__getattribute__(value, "_jinest_source")
                value_context_origin = frame.context_origin_source
                if value_context_origin is None:
                    function_body_path = frame.function_body_source_path
                    function_origin = frame.function_origin_source
                    in_function_body = (
                        function_origin is not None
                        and function_body_path is not None
                        and self._same_source_tree(value_source, function_origin)
                        and value_source.source_path[: len(function_body_path)]
                        == function_body_path
                    )
                    value_context_origin = (
                        function_origin if in_function_body else value_source
                    )
                result = _LayerValue(
                    value_source,
                    frame.local_vars,
                    value_context_origin,
                    layer.hidden,
                )
            elif isinstance(value, Mapping):
                result = _LayerValue(
                    _Source(
                        owner_source.resolver,
                        value,
                        owner_source.source_path + (layer.source_key,),
                        owner_source.document,
                    ),
                    hidden=layer.hidden,
                )
            else:
                self._raise_merge_type_error(
                    bind, layer.source_key, value, "a mapping", source=owner_source
                )
        except Exception:
            cache.pop(cache_key, None)
            raise

        cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # Source views, node metadata, and path operations
    # ------------------------------------------------------------------

    def _raw_source_at(
        self,
        path: tuple[Any, ...],
        document: _SourceDocument | None = None,
    ) -> Any:
        value = self.data if document is None else document.raw
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
        if source.document is not None and not source.source_path:
            return source.resolver._source_root_for(source)
        cache_key = source.instance_id
        cached = source.resolver._source_view_cache.get(cache_key)
        if cached is not None:
            return cached

        parent: _ContainerProxy | None = None
        if source.source_path:
            parent_path = source.source_path[:-1]
            try:
                parent_raw = source.resolver._raw_source_at(parent_path, source.document)
            except (KeyError, IndexError, TypeError, JinestPathError):
                parent_raw = None
            if source.resolver._is_container(parent_raw):
                parent = source.resolver._source_view(
                    _Source(source.resolver, parent_raw, parent_path, source.document)
                )

        proxy = source.resolver._wrap(
            source.raw,
            parent=parent,
            path=source.source_path,
            origin=source.resolver,
            source_path=source.source_path,
            path_kind="source",
            source_document=source.document,
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

    def _invoke_python_function(
        self,
        function: JinestFunction,
        scope: _ContainerProxy,
        source: _Source,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        local_vars: Mapping[str, Any] | None,
    ) -> Any:
        """Invoke a declared function through the same Jinja call machinery."""
        class PythonCallContext:
            def __init__(self) -> None:
                self.vars = {_INTERNAL_SCOPE: scope, _INTERNAL_FUNCTION_LOCALS: local_vars}
                self.parent = dict(self.vars)

            def resolve_or_missing(self, name: str) -> Any:
                return self.vars.get(name, missing)

        return self._invoke_function(function, PythonCallContext(), args, kwargs)

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
        binding_path = call_path
        keypath = self._frame_get(jinja_context, "keypath")
        if isinstance(keypath, PathRef):
            absolute_keypath = keypath._jinest_absolute()
            binding_path = object.__getattribute__(absolute_keypath, "_jinest_segments")
        declaration_path = source.source_path + (spec.source_key,)
        display_declaration = _format_path_segments("root", declaration_path)
        display_call = _format_path_segments("global_root", call_path)
        call_chain = " -> ".join(self._function_stack + [spec.name])
        if spec.mode == "structural" and self._is_recursive_structural_call(
            call_scope,
            source,
            declaration_path,
        ):
            raise JinestFunctionError(
                f"Recursive structural function {spec.name!r} is not supported "
                f"at {display_call}"
            )
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

        inherited_locals = jinja_context.vars.get(
            _INTERNAL_FUNCTION_LOCALS,
            jinja_context.parent.get(_INTERNAL_FUNCTION_LOCALS),
        )
        call_locals = dict(inherited_locals or {})
        call_locals.update(bound)

        marker = "=" if spec.mode == "structural" else _FUNCTION_MODE_MARKERS[spec.mode]
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
                        local_vars=call_locals,
                    )
                    call_locals[parameter.name] = bound[parameter.name]
                except JinestError as exc:
                    raise JinestFunctionError(
                        f"Failed to evaluate default for function {spec.name!r} "
                        f"at {display_declaration}, call site {display_call}: {exc}"
                    ) from exc

            if spec.mode == "structural":
                # The returned proxy is rebound once more by the enclosing
                # field/container.  Keeping this temporary binding separate
                # gives each invocation an isolated parameter cache.
                return self._wrap(
                    spec.template,
                    parent=call_scope,
                    path=binding_path,
                    origin=source.resolver,
                    source_path=declaration_path,
                    source_document=source.document,
                    path_kind=object.__getattribute__(call_scope, "_jinest_path_kind"),
                    local_vars=call_locals,
                    function_scope=call_scope,
                    function_origin_source=source,
                    function_body_source_path=declaration_path,
                )

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
                    local_vars=call_locals,
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
    def _is_recursive_structural_call(
        scope: _ContainerProxy,
        source: _Source,
        declaration_path: tuple[Any, ...],
    ) -> bool:
        """Whether a structural call targets an active structural ancestor."""
        current: _ContainerProxy | None = scope
        while isinstance(current, _ContainerProxy):
            active_origin = object.__getattribute__(
                current, "_jinest_binding"
            ).frame.function_origin_source
            if (
                active_origin is not None
                and active_origin.document_id == source.document_id
                and object.__getattribute__(current, "_jinest_binding").frame.function_body_source_path
                == declaration_path
            ):
                return True
            current = object.__getattribute__(current, "_jinest_parent")
        return False

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

    def _compile_jinja(
        self,
        origin_source: _Source,
        source_key: Any | None,
        kind: EvaluatorKind,
        template: str,
        factory: Any,
    ) -> Any:
        """Cache compiled Jinja artifacts by declaration, kind, and source text."""
        if source_key is None:
            return factory()
        key = (_DeclarationId(origin_source.document_id, source_key), kind, template)
        bridge = origin_source.resolver._jinja
        cache = bridge.compilation_cache
        compiled = cache.get(key, _MISSING)
        if compiled is _MISSING:
            template_key = (kind, template)
            compiled = bridge.template_cache.get(template_key, _MISSING)
            if compiled is _MISSING:
                compiled = factory()
                bridge.template_cache[template_key] = compiled
            cache[key] = compiled
        return compiled

    def _render_plan(self, plan: EvaluationPlan, **kwargs: Any) -> Any:
        """Execute one typed plan through the legacy render primitive."""
        return self._render(
            kwargs.pop("scope"),
            plan.template,
            mode=plan.kind,
            source_key=plan.source_key,
            **kwargs,
        )

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
        prepare_native: bool = True,
        context_origin_source: _Source | None = None,
    ) -> Any:
        if mode not in {"text", "native", "script"}:
            raise ValueError(f"Unsupported render mode: {mode!r}")

        scope_path = object.__getattribute__(scope, "_jinest_path")
        context_scope = scope
        if context_path is None:
            context_path = scope_path
        if local_vars is None:
            local_vars = object.__getattribute__(scope, "_jinest_binding").frame.local_vars
        frame = object.__getattribute__(scope, "_jinest_binding").frame
        function_origin_source = frame.function_origin_source
        function_body_source_path = frame.function_body_source_path
        # A structural function body keeps the declaration's lexical origin,
        # but a node returned by that body may have a different real source
        # (most importantly an imported document). Do not let the function
        # frame replace that node's root/origin/file/import base.
        in_function_body = (
            function_origin_source is not None
            and function_body_source_path is not None
            and self._same_source_tree(origin_source, function_origin_source)
            and origin_source.source_path[: len(function_body_source_path)]
            == function_body_source_path
        )
        if context_origin_source is None:
            context_origin_source = (
                function_origin_source if in_function_body else origin_source
            )
        if source_path is None:
            source_path = (
                origin_source.source_path
                if source_key is None
                else origin_source.source_path + (source_key,)
            )
        if context_path is None:
            context_path = scope_path

        display_path = _format_path_segments("root", source_path)
        if not _valid_evaluator_body(template, mode):
            marker = _FUNCTION_MODE_MARKERS[mode]
            raise JinestTemplateError(
                f"{marker} evaluator {display_path} requires "
                f"{_evaluator_body_requirement(mode)}, "
                f"got {type(template).__name__}",
                path=display_path,
                file=self._source_label_for(origin_source),
            )
        if not isinstance(template, str):
            return self._prepare_native(template)

        path_kind = object.__getattribute__(context_scope, "_jinest_path_kind")
        path_root = (
            self._global_owner.root
            if path_kind == "global"
            else self._source_root_for(context_origin_source)
        )
        path_owner = self if path_kind == "global" else context_origin_source.resolver
        context_path_ref = PathRef(
            path_owner,
            path_root,
            path_kind,
            context_path,
        )
        # `keypath` is the path to the field currently being evaluated.  It is
        # deliberately derived from the same logical key and context path as
        # the other key metadata, so it also works for hidden and overridden
        # declarations.
        keypath = (
            None if keyname is None else context_path_ref[keyname]
        )
        origin_context = context_origin_source.resolver._source_view(
            context_origin_source
        )
        context = {
            _INTERNAL_SCOPE: scope,
            "context": context_scope,
            "origin": origin_context,
            "root": self._source_root_for(context_origin_source),
            "global_root": self._global_owner.root,
            "_": object.__getattribute__(context_scope, "_jinest_parent"),
            "path": context_path_ref,
            "keyname": keyname,
            "effective_key": effective_key,
            "keymode": keymode,
            "keypath": keypath,
        }
        context[_INTERNAL_EVALUATOR_CONTEXT] = MappingProxyType(context.copy())
        if local_vars is not None:
            context[_INTERNAL_FUNCTION_LOCALS] = local_vars
            context.update(local_vars)
        environment = (
            context_origin_source.resolver.script_environment
            if mode == "script"
            else context_origin_source.resolver.environment
        )
        try:
            if mode == "native":
                if "{{" in template or "{%" in template or "{#" in template:
                    raise JinestTemplateError(
                        f"Native expression {display_path} must not use "
                        "Jinja template delimiters"
                    )
                expression = self._compile_jinja(
                    origin_source,
                    source_key,
                    EvaluatorKind.NATIVE,
                    template,
                    lambda: environment.compile_expression(
                        template, undefined_to_none=False
                    ),
                )
                result = expression(**context)
                if prepare_native:
                    return self._prepare_native(result)
                if isinstance(result, Undefined):
                    if not self.strict:
                        return None
                    result._fail_with_undefined_error()
                return result

            if mode == "script":
                compiled = self._compile_jinja(
                    origin_source,
                    source_key,
                    EvaluatorKind.SCRIPT,
                    template,
                    lambda: environment.from_string(
                        self._normalize_multiline_returns(template)
                    ),
                )
                try:
                    compiled.render(**context)
                except _ScriptReturn as returned:
                    return self._prepare_native(returned.value)
                return None

            compiled = self._compile_jinja(
                origin_source,
                source_key,
                EvaluatorKind.TEXT,
                template,
                lambda: environment.from_string(template),
            )
            # NativeEnvironment.render() applies literal_eval to a complete
            # textual result. Generate chunks directly so @ always remains text
            # (for example, a template producing ``"hello"`` keeps its quotes).
            return "".join(str(chunk) for chunk in compiled.generate(**context))
        except _ScriptReturn as returned:
            # Defensive fallback in case an environment layer lets the return
            # escape outside the inner render call.
            return self._prepare_native(returned.value)
        except JinestError as exc:
            self._annotate_error(
                exc,
                path=display_path,
                file=self._source_label_for(origin_source),
            )
            raise
        except UndefinedError as exc:
            if not self.strict:
                return "" if mode == "text" else None
            error = JinestTemplateError(
                f"Failed to render {display_path}: {exc}",
                path=display_path,
                file=self._source_label_for(origin_source),
            )
            raise error from exc
        except Exception as exc:
            error = JinestTemplateError(
                f"Failed to render {display_path}: {exc}",
                path=display_path,
                file=self._source_label_for(origin_source),
            )
            raise error from exc

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
                result: dict[Any, Any] = {}
                for key, item in value.items():
                    cloned_key = self._clone_mapping_key(key, active=active)
                    if cloned_key in result:
                        raise JinestError(
                            f"Duplicate mapping key after cloning: {cloned_key!r}"
                        )
                    result[cloned_key] = self._clone_unresolved(item, active=active)
                return result
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

        document = self._documents.get(path, format)
        if document is None:
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
            document = self._documents.put(path, format, data)

        # Import ancestry affects cycle detection, so it is part of the
        # runtime key. Equal occurrences can share the immutable source
        # runtime; destination rebinding still creates fresh binding caches.
        import_chain = self._import_chain + (path,)
        runtime_key = (path, format, import_chain)
        child = self._documents.runtimes.get(runtime_key)
        if child is None:
            child = Resolver(
                document.data,
                strict=self.strict,
                sandboxed=self.sandboxed,
                globals=self._user_globals,
                filters=self._user_filters,
                source_path=path,
                import_roots=self.import_roots,
                function_max_depth=self.function_max_depth,
                emit_messages=False,
                treat_warnings_as_errors=False,
                debug=self.debug,
                stdlib=self.config.stdlib,
                stdlib_exclude=self.config.stdlib_exclude,
                _import_chain=import_chain,
                _tree_import_chain=self._tree_import_chain,
                _global_owner=self._global_owner,
                _documents=self._documents,
                _document_identity=document.identity,
                _copy_input=False,
            )
            self._documents.runtimes[runtime_key] = child
        return child.root

    # ------------------------------------------------------------------
    # Key enumeration and materialization
    # ------------------------------------------------------------------

    def _public_keys(
        self,
        scope: _MappingProxy,
        *,
        context_origin_source: _Source | None = None,
    ) -> list[Any]:
        source = object.__getattribute__(scope, "_jinest_source")
        if context_origin_source is None:
            context_origin_source = object.__getattribute__(
                scope, "_jinest_binding"
            ).frame.context_origin_source
        result: list[Any] = []
        seen: set[Any] = set()
        self._collect_keys(
            source,
            bind=scope,
            result=result,
            seen=seen,
            active=set(),
            local_vars=None,
            context_origin_source=context_origin_source,
        )
        # Key collection discovers possible names without forcing their values.
        # The unified lookup then applies tombstones and visibility masks.
        return [
            key
            for key in result
            if (
                match := self._find_field(
                    source,
                    key,
                    bind=scope,
                    channel="public",
                    context_origin_source=context_origin_source,
                )
            ) is not _MISSING
            and isinstance(match, _FieldMatch)
            and not match.masked
        ]

    def _collect_keys(
        self,
        source: _Source,
        *,
        bind: _MappingProxy,
        result: list[Any],
        seen: set[Any],
        active: set[tuple[int, int, int | None, int | None]],
        local_vars: Mapping[str, Any] | None = None,
        context_origin_source: _Source | None = None,
    ) -> None:
        token = (
            source.instance_id,
            id(local_vars) if local_vars is not None else None,
            (
                context_origin_source.instance_id
                if context_origin_source is not None
                else None
            ),
        )
        if token in active:
            return
        active.add(token)
        try:
            defaults, overrides = self._layer_stack(
                source,
                bind,
                local_vars=local_vars,
                context_origin_source=context_origin_source,
            )

            for layer in defaults:
                layer_value = self._evaluate_layer(
                    bind,
                    source,
                    layer,
                    local_vars=local_vars,
                    context_origin_source=context_origin_source,
                )
                self._collect_keys(
                    layer_value.source,
                    bind=bind,
                    result=result,
                    seen=seen,
                    active=active,
                    local_vars=layer_value.local_vars,
                    context_origin_source=(
                        layer_value.context_origin_source or context_origin_source
                    ),
                )

            for entry in self._mapping_entries(
                source,
                bind,
                local_vars=local_vars,
                context_origin_source=context_origin_source,
            ):
                source_key = entry.key
                if entry.compose:
                    logical = entry.key
                elif not entry.raw and not entry.dynamic:
                    if self._merge_key(source_key):
                        continue
                    if _field_control_key(source_key) is not None:
                        continue
                    if _parse_function_declaration(source_key) is not None:
                        continue
                    template_info = self._template_key(source_key)
                    logical = template_info[0] if template_info else source_key
                else:
                    # Raw and dynamic keys produce a literal final key; their
                    # result is never fed back into Jinest's key grammar.
                    logical = source_key
                concrete = (
                    entry.raw
                    or entry.dynamic
                    or (
                        not entry.compose
                        and self._template_key(entry.source_key) is None
                    )
                )
                self_wrapper = (
                    _parse_self_declaration(source.raw[entry.source_key])
                    if concrete and isinstance(source.raw, Mapping)
                    else None
                )
                if self_wrapper is not None and self_wrapper.mode == "structural":
                    continue
                if logical in seen:
                    continue
                if (
                    not entry.raw
                    and not entry.dynamic
                    and isinstance(logical, str)
                    and logical.startswith(".")
                ):
                    continue
                if logical not in seen:
                    seen.add(logical)
                    result.append(logical)

            for layer in overrides:
                layer_value = self._evaluate_layer(
                    bind,
                    source,
                    layer,
                    local_vars=local_vars,
                    context_origin_source=context_origin_source,
                )
                self._collect_keys(
                    layer_value.source,
                    bind=bind,
                    result=result,
                    seen=seen,
                    active=active,
                    local_vars=layer_value.local_vars,
                    context_origin_source=(
                        layer_value.context_origin_source or context_origin_source
                    ),
                )
        finally:
            active.remove(token)

    def _to_plain_mapping_key(
        self,
        key: Any,
        *,
        state: _MaterializationState,
    ) -> Any:
        plain_key = self._to_plain(key, state=state)
        # ``strict=False`` may coerce unsupported values to None, but mapping
        # keys must never silently collapse into a different valid key.
        if plain_key is None and key is not None:
            raise JinestError(
                "Unsupported mapping key after materialization: "
                f"{type(key).__name__}"
            )
        try:
            hash(plain_key)
        except TypeError as exc:
            raise JinestError(
                "Unsupported mapping key after materialization: "
                f"{type(plain_key).__name__}"
            ) from exc
        return plain_key

    @staticmethod
    def _set_plain_mapping_item(result: dict[Any, Any], key: Any, value: Any) -> None:
        if key in result:
            raise JinestError(f"Duplicate mapping key after materialization: {key!r}")
        result[key] = value

    def _to_plain(self, value: Any, *, state: _MaterializationState) -> Any:
        if isinstance(value, PathRef):
            return str(value)

        if isinstance(value, _ContainerProxy):
            source = object.__getattribute__(value, "_jinest_source")
            binding = object.__getattribute__(value, "_jinest_binding")
            source_raw_identity = (
                source.document_id.document_identity,
                id(source.raw),
            )
            ancestors = state.proxy_raw.get(source_raw_identity, [])
            if binding.identity in state.bindings or (
                ancestors
                and not self._allows_structural_materialization(value, ancestors)
            ):
                raise JinestError(
                    f"Cyclic container reference at "
                    f"{_format_path(object.__getattribute__(value, '_jinest_path'))}"
                )
            state.bindings.add(binding.identity)
            state.proxy_raw.setdefault(source_raw_identity, []).append(value)
            try:
                if isinstance(value, _MappingProxy):
                    result: dict[Any, Any] = {}
                    for key in self._public_keys(value):
                        plain_key = self._to_plain_mapping_key(key, state=state)
                        self._set_plain_mapping_item(
                            result,
                            plain_key,
                            self._to_plain(
                                self._get_public_field(value, key), state=state
                            ),
                        )
                    return result
                return [
                    self._to_plain(value[index], state=state)
                    for index in range(len(value))
                ]
            finally:
                state.bindings.remove(binding.identity)
                state.proxy_raw[source_raw_identity].pop()
                if not state.proxy_raw[source_raw_identity]:
                    del state.proxy_raw[source_raw_identity]

        if isinstance(value, Mapping):
            raw_id = id(value)
            if raw_id in state.plain_raw:
                raise JinestError("Cyclic mapping in materialized result")
            state.plain_raw.add(raw_id)
            try:
                result: dict[Any, Any] = {}
                for key, item in value.items():
                    plain_key = self._to_plain_mapping_key(key, state=state)
                    self._set_plain_mapping_item(
                        result,
                        plain_key,
                        self._to_plain(item, state=state),
                    )
                return result
            finally:
                state.plain_raw.remove(raw_id)

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            raw_id = id(value)
            if raw_id in state.plain_raw:
                raise JinestError("Cyclic sequence in materialized result")
            state.plain_raw.add(raw_id)
            try:
                return [self._to_plain(item, state=state) for item in value]
            finally:
                state.plain_raw.remove(raw_id)

        return self._copy_scalar(value)

    @staticmethod
    def _allows_structural_materialization(
        value: _ContainerProxy,
        ancestors: Sequence[_ContainerProxy],
    ) -> bool:
        """Allow nested structural calls only when their argument frames differ."""
        body_path = object.__getattribute__(value, "_jinest_binding").frame.function_body_source_path
        local_vars = object.__getattribute__(value, "_jinest_binding").frame.local_vars
        if body_path is None or local_vars is None:
            return False
        return all(
            object.__getattribute__(ancestor, "_jinest_binding").frame.function_body_source_path
            is not None
            and object.__getattribute__(ancestor, "_jinest_binding").frame.local_vars is not local_vars
            for ancestor in ancestors
        )

    @staticmethod
    def _hashable_key(key: Any) -> Any:
        try:
            hash(key)
            return key
        except TypeError:
            return (type(key).__name__, repr(key))


# ----------------------------------------------------------------------
# Public helper namespaces (kept in this single module intentionally)
# ----------------------------------------------------------------------


def _api_owner(value: Any = _MISSING, resolver: Resolver | None = None) -> Resolver:
    """Get and validate the runtime for one public node/path operation."""
    inferred: Resolver | None = None
    if isinstance(value, PathRef):
        inferred = object.__getattribute__(value, "_jinest_owner")
    elif isinstance(value, _ContainerProxy):
        inferred = object.__getattribute__(value, "_jinest_owner")

    if resolver is not None:
        if not isinstance(resolver, Resolver):
            raise TypeError("resolver must be a Resolver")
        if inferred is not None and inferred._global_owner is not resolver._global_owner:
            raise JinestError("value and resolver must belong to the same Jinest tree")
        return resolver
    if inferred is not None:
        return inferred
    raise TypeError(
        "resolver is required when it cannot be inferred from a Jinest node or PathRef"
    )


def _api_owner_from(*values: Any, resolver: Resolver | None = None) -> Resolver:
    """Infer a resolver and reject contradictory explicit ownership."""
    if resolver is not None:
        owner = _api_owner(resolver=resolver)
        for value in values:
            if value is not _MISSING and isinstance(
                value, (PathRef, _ContainerProxy)
            ):
                _api_owner(value, owner)
        return owner
    for value in values:
        if value is not _MISSING and isinstance(value, (PathRef, _ContainerProxy)):
            return _api_owner(value)
    raise TypeError(
        "resolver is required when it cannot be inferred from a Jinest node or PathRef"
    )

def _api_materialize(value: Any, resolver: Resolver | None = None) -> Any:
    if isinstance(value, (PathRef, _ContainerProxy)):
        return _api_owner(value, resolver).resolve(value)
    return value


def _literal_tree(value: Any, active: set[int] | None = None) -> Any:
    """Recursively escape ordinary data for a future Jinest parse."""
    if active is None:
        active = set()
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise JinestError("Cyclic mapping cannot be converted by literal()")
        active.add(marker)
        try:
            return {
                (key + "`") if isinstance(key, str) else key: _literal_tree(item, active)
                for key, item in value.items()
            }
        finally:
            active.remove(marker)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        marker = id(value)
        if marker in active:
            raise JinestError("Cyclic sequence cannot be converted by literal()")
        active.add(marker)
        try:
            return [_literal_tree(item, active) for item in value]
        finally:
            active.remove(marker)
    if isinstance(value, str):
        index = 0
        while index < len(value) and value[index] == "`":
            index += 1
        if index + 1 < len(value) and value[index] == "=" and value[index + 1] in _FUNCTION_MODES:
            return "`" + value
    return value


def _api_path_of(value: Any, *, source: bool = False) -> PathRef:
    owner = _api_owner(value)
    return owner._path_of(value, source=source)


def _api_metadata_node(value: Any) -> _ContainerProxy | None:
    """Find the nearest node carrying source metadata for a public value."""
    if isinstance(value, _ContainerProxy):
        return value
    if not isinstance(value, PathRef):
        return None

    owner = _api_owner(value)
    current = value._jinest_absolute()
    while True:
        try:
            target = owner._at_path(current)
        except Exception:
            # Metadata is anchored to a source container. A final lazy scalar
            # need not be resolved merely to discover that anchor; if it fails,
            # walk to its nearest already-addressable parent instead.
            target = _MISSING
        if isinstance(target, _ContainerProxy):
            return target
        segments = object.__getattribute__(current, "_jinest_segments")
        if not segments:
            return None
        current = PathRef(
            owner,
            object.__getattribute__(current, "_jinest_root"),
            object.__getattribute__(current, "_jinest_kind"),
            segments[:-1],
        )


def _api_source_runtime(value: Any, resolver: Resolver | None = None) -> Resolver:
    node = _api_metadata_node(value)
    if node is not None:
        source_owner = object.__getattribute__(node, "_jinest_source").resolver
        if (
            resolver is not None
            and source_owner._global_owner is not resolver._global_owner
        ):
            raise JinestError("value and resolver must belong to the same Jinest tree")
        return source_owner
    return _api_owner(value, resolver)


def _api_source_dir(value: Any, *, resolver: Resolver | None = None) -> str | None:
    return str(_api_source_runtime(value, resolver).base_dir)


def _api_file_path(value: Any, *, anchor: Any = _MISSING, resolver: Resolver | None = None) -> Path:
    if anchor is _MISSING:
        owner = _api_owner_from(value, resolver=resolver)
    else:
        owner = _api_source_runtime(anchor, resolver)
        if resolver is not None and owner._global_owner is not resolver._global_owner:
            raise JinestPathError("anchor and resolver belong to different Jinest trees")
    # A source resolver's base_dir follows its real source, including the
    # documented base_dir fallback for virtual import_tree identities.
    base = owner.base_dir
    path = Path(os.fspath(value)).expanduser()
    path = path if path.is_absolute() else base / path
    path = path.resolve()
    if owner.import_roots is not None and not any(path.is_relative_to(root) for root in owner.import_roots):
        raise JinestImportError(f"Import path is outside permitted roots: {value}")
    return path


def _api_read_text(path: Any, *, anchor: Any = _MISSING, resolver: Resolver | None = None, encoding: str = "utf-8") -> str:
    return _api_file_path(path, anchor=anchor, resolver=resolver).read_text(encoding=encoding)


def _api_read_lines(
    path: Any,
    *,
    anchor: Any = _MISSING,
    resolver: Resolver | None = None,
    encoding: str = "utf-8",
    keepends: bool = False,
) -> list[str]:
    return _api_read_text(path, anchor=anchor, resolver=resolver, encoding=encoding).splitlines(
        keepends=keepends
    )


def _api_read_bytes(
    path: Any, *, anchor: Any = _MISSING, resolver: Resolver | None = None
) -> bytes:
    return _api_file_path(path, anchor=anchor, resolver=resolver).read_bytes()


def _api_load_document(
    path: Any,
    format: str,
    *,
    anchor: Any = _MISSING,
    resolver: Resolver | None = None,
    encoding: str = "utf-8",
) -> Any:
    """Read one ordinary JSON/YAML document through the common file policy.

    A bare Python helper call retains its useful standalone form.  Calls tied
    to a Resolver/node/path use the same source-relative canonical path and
    import-root checks as imports and Jinja file helpers.
    """
    if resolver is None and anchor is _MISSING:
        text = Path(os.fspath(path)).read_text(encoding=encoding)
    else:
        text = _api_read_text(path, anchor=anchor, resolver=resolver, encoding=encoding)
    return SerializationCodecs.parse(text, format)


def _api_import(path: Any, format: str, *, resolver: Resolver | None = None, anchor: Any = _MISSING) -> Any:
    if anchor is _MISSING:
        owner = _api_owner(resolver=resolver)
    else:
        owner = _api_source_runtime(anchor, resolver)
        if resolver is not None and owner._global_owner is not resolver._global_owner:
            raise JinestPathError("anchor and resolver belong to different Jinest trees")
    return owner._import_tree(path, format)


def _api_serialize(value: Any, *, format: str = "json", file: str | os.PathLike[str] | None = None, resolver: Resolver | None = None) -> str:
    text = SerializationCodecs.serialize(_api_materialize(value, resolver), format.lower())
    if file is not None:
        # The file is exactly the serialized representation returned below.
        # This makes serialization byte-for-byte reproducible.
        Path(file).write_text(text, encoding="utf-8")
    return text


def _ordered_union(*values: Any) -> list[Any]:
    result: list[Any] = []
    for group in values:
        for item in group:
            if not any(item == existing for existing in result): result.append(item)
    return result


def _ordered_intersect(first: Any, *rest: Any) -> list[Any]:
    result: list[Any] = []
    for item in first:
        if all(any(item == candidate for candidate in group) for group in rest) and not any(item == existing for existing in result):
            result.append(item)
    return result


def _ordered_filter(first: Any, rest: Any, *, include: bool) -> list[Any]:
    result=[]
    for item in first:
        present=any(item == candidate for candidate in rest)
        if present is include and not any(item == old for old in result): result.append(item)
    return result


def _combine(
    a: Mapping[Any, Any],
    b: Mapping[Any, Any],
    recursive: bool = False,
) -> dict[Any, Any]:
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        raise TypeError("combine() requires two mappings")
    if not isinstance(recursive, bool):
        raise TypeError("combine recursive must be a boolean")

    # Establish deterministic first-mapping key order without resolving its
    # values. Values shadowed by ``b`` must remain lazy and untouched.
    result = {key: _MISSING for key in a}
    for key, value in b.items():
        # A non-mapping value from ``b`` replaces the old value outright.
        # Do not touch the shadowed lazy value merely because recursive mode
        # is enabled: it cannot participate in a recursive mapping merge.
        if recursive and key in result and isinstance(value, Mapping):
            previous = a[key]
            if isinstance(previous, Mapping):
                value = _combine(previous, value, recursive=True)
        result[key] = value
    for key, value in tuple(result.items()):
        if value is _MISSING:
            result[key] = a[key]
    return result


def _api_root_of(value: Any) -> Any:
    owner = _api_owner(value)
    if isinstance(value, PathRef):
        return object.__getattribute__(value._jinest_absolute(), "_jinest_root")
    return owner._root_of(value)


def _api_source_file(value: Any) -> str | None:
    node = _api_metadata_node(value)
    if node is not None:
        source = object.__getattribute__(node, "_jinest_source")
        return source.resolver._source_label_for(source)
    return _api_owner(value)._source_label


def _runtime_owner(resolver: Resolver | None, kwargs: Mapping[str, Any]) -> Resolver:
    return _api_owner_from(
        kwargs.get("context", _MISSING), kwargs.get("origin", _MISSING),
        kwargs.get("root", _MISSING), resolver=resolver,
    )


helpers = SimpleNamespace(
    path=SimpleNamespace(
        normalize_path=lambda value, *, resolver=None, anchor=_MISSING: _api_owner_from(value, anchor, resolver=resolver).normalize_path(value, anchor=anchor),
        absolute_path=lambda value, *, resolver=None, anchor=_MISSING: _api_owner_from(value, anchor, resolver=resolver).absolute_path(value, anchor=anchor),
        relative_path=lambda target, *, resolver=None, base=_MISSING: _api_owner_from(target, base, resolver=resolver).relative_path(target, base=base),
        path_of=lambda value: _api_path_of(value), source_path_of=lambda value: _api_path_of(value, source=True),
        at=lambda target, *, resolver=None, anchor=_MISSING: _api_owner_from(target, anchor, resolver=resolver).at(target, anchor=anchor),
        get=lambda target, default=None, *, resolver=None, anchor=_MISSING: _api_owner_from(target, anchor, resolver=resolver).get(target, default, anchor=anchor),
        root_of=_api_root_of, source_file=_api_source_file, source_dir=_api_source_dir,
    ),
    files=SimpleNamespace(file_path=_api_file_path, read_text=_api_read_text, read_lines=_api_read_lines, read_bytes=_api_read_bytes, file_exists=lambda path, **kwargs: _api_file_path(path, **kwargs).exists()),
    runtime=SimpleNamespace(node=lambda value, *, resolver=None, **kwargs: _runtime_owner(resolver, kwargs).node(value, **kwargs), resolve=lambda value, *, resolver=None, **kwargs: _api_owner_from(value, kwargs.get("context", _MISSING), resolver=resolver).resolve(value, **kwargs), eval=lambda expression, *, resolver=None, **kwargs: _runtime_owner(resolver, kwargs).eval(expression, **kwargs), render=lambda template, *, resolver=None, **kwargs: _runtime_owner(resolver, kwargs).render(template, **kwargs), script=lambda source, *, resolver=None, **kwargs: _runtime_owner(resolver, kwargs).script(source, **kwargs), literal=_literal_tree),
    documents=SimpleNamespace(load_json=lambda path, **kwargs: _api_load_document(path, "json", **kwargs), load_yaml=lambda path, **kwargs: _api_load_document(path, "yaml", **kwargs), import_json=lambda path, *, resolver=None, anchor=_MISSING: _api_import(path, "json", resolver=resolver, anchor=anchor), import_yaml=lambda path, *, resolver=None, anchor=_MISSING: _api_import(path, "yaml", resolver=resolver, anchor=anchor), import_tree=lambda value, *, resolver=None, **kwargs: _runtime_owner(resolver, kwargs).import_tree(value, **kwargs), export_json=lambda value, path, **kwargs: _api_serialize(value, format="json", file=path, **kwargs), export_yaml=lambda value, path, **kwargs: _api_serialize(value, format="yaml", file=path, **kwargs)),
    serialization=SimpleNamespace(from_json=lambda text: SerializationCodecs.parse(text, "json"), from_yaml=lambda text: SerializationCodecs.parse(text, "yaml"), to_json=lambda value, **kwargs: _api_serialize(value, format="json", **kwargs), to_yaml=lambda value, **kwargs: _api_serialize(value, format="yaml", **kwargs), json_normalize=lambda value: _normalize_json_value(_api_materialize(value), active=set()), yaml_normalize=lambda value: _normalize_yaml_value(_api_materialize(value), active=set()), serialize=_api_serialize),
    collections=SimpleNamespace(combine=_combine, union=_ordered_union, intersect=lambda first, *rest: _ordered_intersect(first, *rest), difference=lambda first, *rest: _ordered_filter(first, _ordered_union(*rest), include=False), symmetric_difference=lambda a, b: _ordered_union(_ordered_filter(a, b, include=False), _ordered_filter(b, a, include=False))),
)

# Direct namespace imports are supported by the standalone module. Wheel-only
# ``jinest.helpers.<namespace>`` modules re-export these same objects.
path = helpers.path
files = helpers.files
runtime = helpers.runtime
documents = helpers.documents
serialization = helpers.serialization
collections = helpers.collections


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

    data = SerializationCodecs.parse(text, source_format)

    result = Resolver(data, **resolver_options).resolve()
    return SerializationCodecs.serialize(result, target_format)


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
    data = SerializationCodecs.parse(input_path.read_text(encoding="utf-8"), source_format)

    # The source path is essential for relative imports.
    resolver_options.setdefault("source_path", input_path)
    result = Resolver(data, **resolver_options).resolve()
    target_format = (output_format or source_format).lower()
    rendered = SerializationCodecs.serialize(result, target_format)

    if output is not None:
        Path(output).write_text(
            rendered + ("" if rendered.endswith("\n") else "\n"),
            encoding="utf-8",
        )
    return rendered


def _parse_text(text: str, format: str) -> Any:
    if format == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise JinestError(f"Invalid JSON input: {exc.msg}") from exc
    if format in {"yaml", "yml"}:
        yaml = _import_yaml_module()
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            detail = getattr(exc, "problem", None) or str(exc)
            raise JinestError(f"Invalid YAML input: {detail}") from exc
    raise ValueError(f"Unsupported input format: {format!r}")


def _serialize(value: Any, format: str) -> str:
    if format == "json":
        return json.dumps(
            _normalize_json_value(value, active=set()),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    if format in {"yaml", "yml"}:
        return _import_yaml_module().safe_dump(
            _normalize_yaml_value(value, active=set()),
            allow_unicode=True,
            sort_keys=False,
        )
    raise ValueError(f"Unsupported output format: {format!r}")


def _normalize_yaml_value(value: Any, *, active: set[int]) -> Any:
    """Convert accepted extended scalars to values representable by SafeDumper."""
    if isinstance(value, PathRef):
        return str(value)
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, Mapping):
        value_id = id(value)
        if value_id in active:
            raise JinestError("Cyclic mapping cannot be serialized as YAML")
        active.add(value_id)
        try:
            result: dict[Any, Any] = {}
            for key, item in value.items():
                normalized_key = _normalize_yaml_value(key, active=active)
                try:
                    duplicate = normalized_key in result
                except TypeError as exc:
                    raise JinestError(
                        "YAML mapping keys must be hashable after normalization"
                    ) from exc
                if duplicate:
                    raise JinestError(
                        "Duplicate YAML mapping key after normalization: "
                        f"{normalized_key!r}"
                    )
                result[normalized_key] = _normalize_yaml_value(item, active=active)
            return result
        finally:
            active.remove(value_id)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        value_id = id(value)
        if value_id in active:
            raise JinestError("Cyclic sequence cannot be serialized as YAML")
        active.add(value_id)
        try:
            return [_normalize_yaml_value(item, active=active) for item in value]
        finally:
            active.remove(value_id)
    return value


def _normalize_json_value(value: Any, *, active: set[int]) -> Any:
    """Convert extended scalar values to standards-compliant JSON."""
    if isinstance(value, PathRef):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        # Latin-1 maps every byte 0x00..0xFF to the matching Unicode code point.
        # json.loads(...).encode("latin-1") therefore reconstructs the bytes.
        return bytes(value).decode("latin-1")
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
            result: dict[Any, Any] = {}
            json_keys: set[str] = set()
            for key, item in value.items():
                normalized_key = _normalize_json_value(key, active=active)
                if normalized_key is None:
                    json_key = "null"
                elif isinstance(normalized_key, bool):
                    json_key = "true" if normalized_key else "false"
                elif isinstance(normalized_key, str):
                    json_key = str(normalized_key)
                elif isinstance(normalized_key, (int, float)):
                    json_key = str(normalized_key)
                else:
                    raise JinestError(
                        "JSON object keys must normalize to a scalar, got "
                        f"{type(normalized_key).__name__} from {key!r}"
                    )
                if json_key in json_keys:
                    raise JinestError(
                        "Duplicate JSON object key after normalization: "
                        f"{json_key!r}"
                    )
                json_keys.add(json_key)
                result[normalized_key] = _normalize_json_value(
                    item,
                    active=active,
                )
            return result
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
    def expect(actual: Any, expected: Any, label: str) -> None:
        if actual != expected:
            raise RuntimeError(
                f"Jinest self-test failed ({label}): "
                f"expected {expected!r}, got {actual!r}"
            )

    data = {
        "defaults1": {"rank": "d1", "d1": True},
        "defaults2": {"rank": "d2", "d2": True},
        "overrides1": {"rank": "o1", "o1": True},
        "overrides2": {"rank": "o2", "o2": True},
        "example": {
            "<<!2$": "root.overrides2",
            "<<2$": "root.defaults2",
            "x": 1,
            "y@": "{{ x }}",
            "z$": "x + (y | int)",
            "rank": "local",
            "<<!1$": "root.overrides1",
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
        "ready_array$": "['var1', 'root.var2']",
        "path_list": [
            {"obj": {"where$": "path"}},
            {"obj": {"where@": "{{ path }}"}},
        ],
    }

    resolver = Resolver(data, emit_messages=False)
    checks = [
        (resolver.root.example.rank, "o2", "override priority"),
        (resolver.root.example.z, 2, "native field"),
        (resolver.root.priority.value, 42, "field priority"),
        (resolver.root.C.x, 1, "prototype chain"),
        (resolver.root["class"].prototype.A, 1, "parent lookup"),
        (resolver.root.inherited.instance.A, 2, "rebound parent"),
        (resolver.root.inherited.instance.B, 1, "rebound local"),
        (resolver.root.inherited.instance.C, 10, "source root"),
        (
            resolver.root.inherited.instance.where,
            "global_root.inherited.instance",
            "rebound path",
        ),
        (
            resolver.root.inherited.other.where,
            "global_root.inherited.other",
            "second rebound path",
        ),
        (resolver.root.native_array[0], 7, "native array local"),
        (resolver.root.native_array[1], 9, "native array root"),
        (
            str(resolver.root.native_array[6]),
            "global_root.native_array[6]",
            "native array path",
        ),
        (
            resolver.root.text_array[4],
            "global_root.text_array[4]",
            "text array path",
        ),
        (resolver.root.ready_array[:], ["var1", "root.var2"], "ready array"),
        (
            str(resolver.root.path_list[0].obj.where),
            "global_root.path_list[0].obj",
            "list object path",
        ),
        (
            str(resolver.root.inherited.instance.path),
            "global_root.inherited.instance",
            "node path",
        ),
        (
            str(resolver.root.inherited.instance.source_path),
            "root.inherited.instance",
            "node source path",
        ),
    ]
    for actual, expected, label in checks:
        expect(actual, expected, label)

    script = Resolver({
        "x": 4,
        "value^": "% set y = x * 2\n% return {'y': y}\n",
        "target": {"<<^": "% return {'a': 1}\n", "b": 2},
    }).resolve()
    expect(script["value"], {"y": 8}, "script field")
    expect(script["target"], {"a": 1, "b": 2}, "script merge")

    result = resolver.resolve()
    expect(result["cycle"], {"a": None, "b": None}, "field cycle")
    expect(result["example"]["rank"], "o2", "materialized priority")
    expect(result["example"]["d1"], True, "first default")
    expect(result["example"]["d2"], True, "second default")
    expect(result["example"]["o1"], True, "first override")
    expect(result["example"]["o2"], True, "second override")

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
        expect(imported["instance"]["absolute"], 10, "import source root")
        expect(imported["instance"]["relative"], 5, "import destination parent")
        expect(
            imported["instance"]["where"],
            "global_root.instance",
            "import destination path",
        )
        expect(imported["json_obj"]["double"], 42, "JSON object import")
        expect(imported["json_value"], 42, "JSON scalar access")
        expect(imported["via_filter"], 10, "import filter")

        # Lazy import cycle: the path already present in the import ancestry is None.
        (folder / "a.yaml").write_text("other$: import('b.yaml')\n", encoding="utf-8")
        (folder / "b.yaml").write_text("back$: import('a.yaml')\n", encoding="utf-8")
        cycle = _parse_text(resolve_file(folder / "a.yaml", output_format="json"), "json")
        expect(cycle, {"other": {"back": None}}, "import cycle")

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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print source file and path for each message or error",
    )
    parser.add_argument("--self-test", action="store_true", help="Run built-in tests")
    args = parser.parse_args()

    if not args.input:
        if not args.self_test:
            parser.error("input is required unless --self-test is used")

    try:
        if args.self_test:
            _self_test()
            return
        rendered = resolve_file(
            args.input,
            output=args.output,
            output_format=args.output_format,
            sandboxed=not args.unsafe,
            emit_messages=not args.no_messages,
            treat_warnings_as_errors=args.treat_warnings_as_errors,
            debug=args.debug,
        )
    except Exception as exc:
        if args.debug and isinstance(exc, JinestError) and getattr(
            exc, "_jinest_debug_emitted", False
        ):
            parser.exit(1)
        details = f"jinest: {exc}\n"
        if args.debug:
            details += f"  at root\n  in {args.input or '<self-test>'}\n"
        parser.exit(1, details)
    if args.output is None:
        sys.stdout.write(rendered)
        if not rendered.endswith("\n"):
            sys.stdout.write("\n")


if __name__ == "__main__":
    _main()
