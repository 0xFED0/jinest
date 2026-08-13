<div align="center">

# Jinest

### Lazy, structured Jinja resolution for Python, JSON, and YAML

**Native expressions. Multiline scripts. Lazy prototype layers. Source-aware paths.**

</div>

---

Jinest turns an ordinary Python object tree into a lazy configuration graph.
Instead of rendering an entire document into text, it resolves individual fields only when they are accessed and preserves mappings, lists, numbers, booleans, and `null` as native values.

```yaml
app:
  host: localhost
  port: 8080

  url@: "http://{{ host }}:{{ port }}/api"
  enabled$: "port > 0"

  metadata^: |
    % set scheme = "https" if port == 443 else "http"
    % return {
      "scheme": scheme,
      "address": host ~ ":" ~ port,
      "path": path
    }
```

Resolves to:

```yaml
app:
  host: localhost
  port: 8080
  url: http://localhost:8080/api
  enabled: true
  metadata:
    scheme: http
    address: localhost:8080
    path: global_root.app
```

> **Status:** Jinest `0.14.1` is a single-file prototype with a standalone regression suite. The public API may still evolve before `1.0`.

## Contents

- [Highlights](#highlights)
- [Installation](#installation)
- [Mental model](#mental-model)
- [Syntax](#syntax)
- [Native expressions: `$`](#native-expressions-)
- [Text templates: `@`](#text-templates-)
- [Multiline scripts: `^`](#multiline-scripts-)
- [Evaluation layers, raw keys, and inline syntax](#evaluation-layers-raw-keys-and-inline-syntax)
- [Evaluation context](#evaluation-context)
- [Hidden fields](#hidden-fields)
- [Diagnostics: warnings and hints](#diagnostics-warnings-and-hints)
- [Template functions](#template-functions)
- [Compose declarations](#compose-declarations)
- [Self-declaration wrappers](#self-declaration-wrappers)
- [Lazy layers](#lazy-layers)
- [Lazy arrays](#lazy-arrays)
- [Cycles](#cycles)
- [Node metadata](#node-metadata)
- [PathRef](#pathref)
- [Path functions](#path-functions)
  - [`path_of(node)`](#path_ofnode)
  - [`source_path_of(node)`](#source_path_ofnode)
  - [`normalize_path(value)`](#normalize_pathvalue)
  - [`absolute_path(value, anchor=context)`](#absolute_pathvalue-anchorcontext)
  - [`relative_path(target, base=context)`](#relative_pathtarget-basecontext)
  - [`at(path, anchor=context)`](#atpath-anchorcontext)
  - [`get(path, default=none, anchor=context)`](#getpath-defaultnone-anchorcontext)
  - [Node indexing](#node-indexing)
  - [Source helpers](#source-helpers)
- [Imports](#imports)
- [Python API](#python-api)
  - [Eager convenience](#eager-convenience)
  - [Lazy access](#lazy-access)
  - [Files and text](#files-and-text)
  - [Resolver options](#resolver-options)
- [CLI](#cli)
- [Testing](#testing)
- [Examples](#examples)
- [Security](#security)
- [License](#license)

## Highlights

- Lazy, memoized field resolution with deterministic materialization.
- Native expressions (`$`), text templates (`@`), and scripts (`^`).
- Inside-out evaluator pipelines through field, inline, and self syntax.
- Safe template functions and Cartesian structural/text composition.
- Lazy default and override layers without eager dictionary merging.
- Destination-aware reusable nodes with isolated binding caches.
- Hidden intermediate fields and source-aware diagnostics.
- Independent source roots for imported YAML and JSON trees.
- Structured `PathRef` objects and explicit navigation helpers.

## Installation

Jinest is distributed as a single Python module and as a wheel. Python 3.10 is
the minimum supported version; development and compatibility testing primarily
target Python 3.11 and newer.

The packaged distribution deliberately installs both Jinja2 and PyYAML:

```bash
python -m pip install jinest
```

For a local checkout:

```bash
python -m pip install build
python -m build --wheel
python -m pip install dist/jinest-*-py3-none-any.whl
```

The wheel exposes the `jinest` CLI:

```bash
jinest config.yaml
```

When copying `jinest.py` directly, Jinja2 is required for evaluation. PyYAML
remains optional: it is imported only for YAML input or output, so Python-object
and JSON workflows work without it.

## Mental model

Jinest treats a tree as a graph of lazily bound mapping/list nodes. A field is
resolved on first access and memoized within that binding. `resolve()` walks
the public graph and materializes ordinary Python values.

The following mappings and lists are Jinest code:

- containers in the source document;
- containers returned by `$` expressions or `^` scripts;
- structural-function and structural-compose bodies;
- an existing lazy node returned from another source or binding.

Thus a generated `{"value$": "1 + 1"}` becomes `{"value": 2}`. Escape a
generated key with a trailing backtick when it must remain literal:
``{"value$`": "1 + 1"}`` produces the literal key `value$`. Generated inline
strings use the same leading-backtick escape as source strings.

Returning an existing lazy node creates a fresh destination binding with fresh
field, child, key-index, and layer caches. Its declaration source, `origin`,
`root`, and function/compose locals are preserved. The same node can therefore
be attached repeatedly with independent destination `context` and `path`.

Function declarations and hidden helpers participate in lookup but are omitted
from materialized output. Compose declarations emit only their composed value.

## Syntax

Marker position is part of the grammar: a key suffix declares a field or
structural construct; a scalar prefix declares an inline evaluator.

| Form | Meaning |
|---|---|
| `name$`, `name@`, `name^` | Native, text, or script field |
| `name(args)$`, `name(args)@`, `name(args)^` | Scalar template function |
| `name(args)=` | Structural function |
| `name[axis=source, ...]=` | Structural Cartesian compose |
| `name[axis=source, ...]@` | Text Cartesian compose |
| `=$expr`, `=@text`, `=^script` | Inline evaluator in a scalar value or mapping key |
| `<$`, `<(args)=`, `<[axis=source]=` | Declaration applied to the current slot |
| `key\`` | Raw key: remove one final backtick and disable key parsing |
| `.name`, `.name$`, `.name@`, `.name^` | Hidden field |
| `<<$`, `<<N$`, `<<^`, `<<N^` | Default layer |
| `<<!$`, `<<N!$`, `<<!^`, `<<N!^` | Override layer |

Local field priority inside one mapping is:

```text
name > name^ > name$ > name@
```

Lower-priority alternatives are neither parsed nor evaluated. Functions,
compose declarations, dynamic/raw final keys, and self structural declarations
share the logical-key namespace and reject ambiguous duplicates.

YAML quoting still applies. Quote keys and inline strings containing `:`, `#`,
braces, or leading marker characters when YAML could interpret them first.

## Native expressions: `$`

A `$` evaluator normally receives one Jinja expression as a string, without
`{{ ... }}`. Its result preserves native type:

```yaml
values:
  x: 2
  y: 3
  sum$: "x + y"
  enabled$: "sum > 4"
  generated$: "{'answer$': '40 + 2'}"
```

The generated mapping is a Jinest subtree, so `answer$` becomes `answer: 42`.

For convenience, a finite number or boolean body passes through unchanged:

```yaml
attempts$: 3
enabled$: true
```

Mappings, `null`, dates, bytes, and other non-string bodies are invalid. A
direct list has the legacy mode-array semantics described below. Return a
structure or another scalar from a string expression.

## Text templates: `@`

An `@` evaluator always receives a string containing a complete Jinja template,
and its result is always text:

```yaml
release:
  product: Jinest
  version: 0.14.1
  label@: "{{ product }} v{{ version }}"
```

`@` does not accept numeric or boolean passthrough bodies. Write `"42"` or
`"{{ 42 }}"` when text is intended.

## Multiline scripts: `^`

A `^` evaluator normally receives a string containing Jinja line statements
prefixed with `%`. `return` produces a native value:

```yaml
result^: |
  % set doubled = (x + y) * 2
  % if doubled > 10
    % return {"value": doubled, "large": true}
  % endif
  % return {"value": doubled, "large": false}
```

Rules:

- `return expression` immediately terminates the script.
- `return` without an expression, or reaching the end, returns `null`.
- `return` works inside `if`, `for`, and nested Jinja blocks.
- Standard `{% ... %}` blocks remain accepted.
- Returned mappings/lists become Jinest subtrees.
- Numeric and boolean bodies pass through; other non-string bodies are invalid.

A direct list uses legacy mode-array semantics rather than being one script
body.

## Evaluation layers, raw keys, and inline syntax

These forms declare one native evaluation layer:

```yaml
suffix$: "base + 1"
inline: "=$base + 1"
wrapped:
  <$: "base + 1"
```

Inline directives are scalar prefixes. A leading backtick escapes one:

```yaml
native: "=$base * 2"
label: "=@base={{ base }}"
script: "=^% return base * 3"
literal: "`=$base"       # literal string "=$base"
```

Layers compose inside-out. A field/array mode is an implicit outer layer;
inline and nested `<$` forms are inner layers:

```yaml
pipeline$:
  <$: '"base + 1"'
```

Every layer validates the previous result. `@` requires a string;
`$`/`^` accept a string or numeric/boolean passthrough. A mapping/list between
evaluator stages is an error.

A key ending in a backtick is raw. One marker is removed and the remainder is
never reinterpreted:

```yaml
"price$`": literal key named "price$"
"final``": literal key named "final`"
".visible`": literal visible key named ".visible"
```

An inline directive in a mapping key creates a dynamic key. It is evaluated
once when that destination mapping is first indexed, must return a string, and
then becomes a literal final key:

```yaml
name: generated$
"=$name": "=$base + 1"       # literal final key "generated$"
"=$'.visible'": shown        # visible final key ".visible"
```

A raw/dynamic leading dot is not hidden. Duplicate final or logical keys raise
`JinestError`.

## Evaluation context

Every evaluator receives:

| Name | Meaning |
|---|---|
| `context` | Destination mapping/list scope currently evaluating the value |
| `_` | Parent of `context` |
| `path` | Destination `PathRef` for that scope, or for the current array item |
| `origin` | Source mapping/list containing the winning declaration |
| `root` | Root of the source tree owning `origin` |
| `global_root` | Top-level destination root of the original `Resolver` |
| `keyname` | Logical output key; hidden prefixes and field suffixes are removed |
| `effective_key` | Exact source key introducing the active declaration |
| `keymode` | Active marker: `$`, `@`, `^`, `=`, or `none` |
| `keypath` | Exact alias for `path[keyname]`, or `none` without a `keyname` |

Each evaluator in a multi-stage pipeline sees its own `keymode`. Structural
functions/compose use `=` and text compose uses `@`. During dynamic-key
evaluation, `keyname` and `keypath` are `none` because the final key does not
yet exist.

`keypath` is deliberately a syntactic alias. For an item of `items$`, `path`
already includes the array index, so the alias is `path["items"]`; it need not
equal the enclosing declaration path.

Function parameters and compose-axis locals have priority over context fields.

Imported prototypes keep source-absolute references while adapting destination
references:

```yaml
# library.yaml
constant: 10
prototype:
  absolute$: root.constant
  relative$: _.parent_value
  destination@: "{{ context.path }}"
  declaration@: "{{ origin.path }}"
```

```yaml
# main.yaml
parent_value: 5
instance:
  <<$: import("library.yaml").prototype
```

The imported fields see:

```text
context      = global_root.instance
origin       = root.prototype
root         = root of library.yaml
global_root  = root of main.yaml
```

For `value$`, metadata is `keyname == "value"`,
`effective_key == "value$"`, and `keymode == "$"`. Hidden `.value@` keeps
`keyname == "value"` and `effective_key == ".value@"`.

## Hidden fields

Prefix a statically declared field with `.` to create an intermediate value. It
is available to evaluators without the prefix and omitted from materialized
JSON/YAML:

```yaml
price: 100
.tax$: "price * 0.2"
tax: public text
total$: "price + tax"
```

When both `.name` and `name` exist, ordinary Jinest lookup of `name` resolves
the hidden declaration. The public declaration remains independent and is used
for final materialization; it is not reachable through normal lookup while the
hidden declaration is in scope.

Hidden behavior applies to static concrete, `$`, `@`, and `^` fields. A raw key
such as ``".name`"``, or a dynamic key whose result is `.name`, is a visible
literal final key and is never converted into a hidden field.

## Diagnostics: warnings and hints

During schema discovery Jinest collects non-fatal diagnostics in
`Resolver.messages`. Every immutable `JinestMessage` has `level` (`"warning"`
or `"hint"`), `msg`, `path`, and `file`.

A concrete unsuffixed field suppresses lower-priority field modes with the same
logical name; every suppressed declaration gets a warning. If both `key` and
`.key` exist, Jinest adds a hint because hidden lookup differs from final
materialization.

By default messages are printed to stderr after successful resolution:

```text
jinest: warning: ...
  at root.path
  in /project/config.yml
```

The `at`/`in` lines are included only with `debug=True`. Locations remain
available programmatically regardless of that flag.

Use `emit_messages=False` to process the list without stderr output.
`treat_warnings_as_errors=True` raises `JinestWarningError` when a warning
exists; hints do not trigger it. CLI equivalents are `--no-messages`
(`-silent`), `--treat-warnings-as-errors` (`-Werror`), and `--debug`.

## Template functions

Mappings declare functions with an evaluator or structural suffix:

```yaml
square(x)$: "x * x"
quote(value, mark='"')@: "{{ mark }}{{ value }}{{ mark }}"
clamp(value, minimum, maximum)^: |
  % if value < minimum
    % return minimum
  % endif
  % return value
```

Scalar function bodies follow ordinary evaluator type rules: `@` requires a
string; `$`/`^` require a string or numeric/boolean passthrough. Functions
support positional, named, and default arguments. `*args`, `**kwargs`,
positional-only, keyword-only, and annotated parameters are rejected.

Defaults are native expressions evaluated at call time. Earlier parameters are
visible to later defaults, and parameters take priority over context fields:

```yaml
defaults:
  factor: 10
scale(value, factor=global_root.defaults.factor)$: "value * factor"
result$: "scale(5)"
```

Function declarations are lazy helpers and omitted from output. A mapping
containing only functions materializes as an empty mapping. Scalar functions
use call-site `context`, `path`, and `global_root`; `origin` and `root` refer
to the declaration source.

Structural functions end in `=` and require a mapping or list body:

```yaml
record(x, key, value)=:
  value$: x
  =$key: =$value

pair(a, b)=:
  - =$a
  - =$b

result: "=$record(2, 'answer', 42)"
values: "=$pair('left', 'right')"
```

Each call creates an isolated parameter frame and temporary lazy node. Every
destination attachment creates a fresh ordinary binding and fresh caches.
Destination `context`/`path` are derived normally; source metadata and function
locals are preserved. Reusing one returned node at multiple destinations
therefore produces independent paths.

Scalar structural bodies and unsuffixed structural-looking declarations such
as `name(args): {}` are errors. Functions, fields, compose declarations, and
dynamic/self declarations cannot claim the same logical name.
`function_max_depth` limits recursive scalar (`$`, `@`, `^`) function calls.
Structural functions are deliberately non-recursive: a direct or indirect
recursive structural call raises `JinestFunctionError`.

## Compose declarations

Compose expands a structural or text body over a Cartesian product:

```yaml
versions: ["3.10", "3.11"]
dirs: [bin, lib]
prefix: run

items[v=versions, d=dirs]=:
  - "=@{{ prefix }}/{{ d }}/py{{ v }}"
  - "=$prefix ~ ':' ~ d ~ ':' ~ v"

summary[v=versions]@: "py{{ v }};"
```

Every `axis=source` is an independent native expression evaluated before the
product. Axis sources cannot reference earlier axes. Each source must be
iterable; lists/tuples are conventional, strings iterate characters, and
mappings iterate keys.

Axes preserve declaration order: the first is outermost and the last
innermost. Axis locals take priority over context fields. An empty axis emits
an empty list/mapping for structural compose and an empty string for text
compose.

For each combination:

- a list body contributes all items with one-level flattening;
- a mapping body contributes all generated entries;
- a text body is rendered once and concatenated in product order.

Mapping bodies support dynamic keys and reject duplicate final keys. Every
generated structural node is freshly rebound at its emitted destination.
Compose declarations are not callable helpers; only their final logical name
is emitted.

## Self-declaration wrappers

A mapping containing exactly one supported `<...` key applies that declaration
to its own value slot. The wrapper key is never materialized:

```yaml
value:
  <$: "base + 1"

record:
  <(x)=:
    value$: x

items:
  <[item=values]=:
    - "=@{{ item }}"
```

Supported forms are:

- `<$`: one native evaluator layer;
- `<(args)=`: structural function at the current slot;
- `<[axis=source]=`: structural compose at the current slot.

`<@` and `<^` are not currently implemented. `<$` follows the same scalar
validation as `$` and may be nested for inside-out pipelines. Structural self
bodies follow normal function/compose validation and rebinding.

A wrapper is recognized only when its mapping has exactly one supported key.
With any additional key, the mapping is ordinary Jinest data. Self declarations
also work under raw or dynamic destination keys.

## Lazy layers

Mappings may inherit lazy default and override layers:

```yaml
defaults:
  host: localhost
  port: 8000
overrides:
  port: 443

service:
  <<$: root.defaults
  port: 8080
  <<!^: |
    % if force_secure
      % return root.overrides
    % endif
    % return null
```

Materialization order is:

```text
defaults -> local fields -> overrides
```

Lookup runs from highest to lowest precedence:

```text
last override -> first override -> local -> last default -> first default
```

`$` and `^` select only how the layer source is computed. Their body
validation follows ordinary evaluator rules; the result must be a mapping or
`null`. `null` means an empty layer.

Numbered keys use `N` as order; omitted `N` is `0`. Default and override
families are sorted independently. Larger `N` has higher lookup priority; at
equal `N`, the later source declaration wins.

## Lazy arrays

Lists are lazy nodes and indices participate in destination paths.

A list directly under `$`, `@`, or `^` is the legacy mode-typed array form:
the list is a container and every item is one body of that mode.

```yaml
native_values$:
  - "x + 1"
  - 123
  - true

text_values@:
  - "Value: {{ x }}"
  - "42"

script_values^:
  - "% return x * 2"
  - 123
  - false
```

For `$`/`^`, items must be strings, numbers, or booleans. For `@`, every item
must be a string. `null`, mappings, and nested lists are invalid mode bodies.

Inline/self syntax inside a mode-typed array is an inner layer and the array
mode is outer. An escaped inline string remains literal and bypasses the outer
mode.

A list returned by an expression/script is a generated Jinest subtree, not a
second legacy mode-typed body. Ordinary returned strings are not implicitly
evaluated again, but explicit `=<mode>` strings and Jinest mapping keys inside
that returned container are parsed normally. Escape them when they are literal.

Example path:

```text
global_root.items[5].object
```

## Cycles

Jinest distinguishes evaluation cycles from physically cyclic Python data:

- a field expression/script cycle resolves the recursive field to `null`;
- a recursively requested merge layer is temporarily an empty mapping;
- an import already active in the current ancestry resolves to `null`;
- recursive scalar functions beyond `function_max_depth` raise `JinestFunctionError`;
- direct and indirect recursive structural-function calls raise `JinestFunctionError`;
- a physically cyclic mapping/list reaching cloning or materialization raises `JinestError`.

Field/layer/import cycle behavior is independent of `strict`. Non-strict mode
controls unsupported scalars and undefined template values, not physical
container cycles.

## Node metadata

Every lazy mapping/list node exposes explicit metadata attributes:

```jinja
node.path
node.source_path
node.root
node.file
```

| Attribute | Meaning |
|---|---|
| `path` | Destination path; normally starts with `global_root` |
| `source_path` | Declaration path inside the source tree; starts with `root` |
| `root` | Source root node |
| `file` | Absolute source filename, or `none` for in-memory data |

Metadata attributes take priority during dot access. A real field with the same name remains available through brackets:

```jinja
object.path       {# metadata #}
object["path"]    {# real field #}
```

No dunder metadata names are used.

## PathRef

`path`, `node.path`, and `node.source_path` are immutable `PathRef` objects.
They render as Jinja-compatible paths but remain navigable before conversion to text:

```jinja
path
path._
path._._.settings
path.items[5].name
path["not-an-identifier"]
path._._.settings.absolute
```

Destination paths render from `global_root`:

```text
global_root.services[5].endpoint
```

Source paths render from the owning source `root`:

```text
root.prototype.settings
```

`PathRef` does not expose node metadata. For example, `path.file` addresses a field named `file`; use `at(path).file` to read metadata from the target node. The path operation name `absolute` can be addressed as a real key with `path["absolute"]`.

When a `PathRef` reaches final JSON/YAML materialization, it is serialized as its canonical string.

## Path functions

### `path_of(node)`

Return the destination path of a mapping/list node.

```jinja
path_of(context)
path_of(global_root.services).api
```

### `source_path_of(node)`

Return the source declaration path of a node.

```jinja
source_path_of(origin)
```

### `normalize_path(value)`

Parse or normalize a path string, `PathRef`, or node:

```jinja
normalize_path("global_root.services[0].name")
normalize_path("_._.settings")
```

### `absolute_path(value, anchor=context)`

Convert a relative path to an absolute path. `root=` is accepted as an alias for `anchor=`.

```jinja
absolute_path(relative, anchor=context)
absolute_path(relative, root=context)
```

### `relative_path(target, base=context)`

Build a relative path. `path=` is accepted as an alias for `base=`.

```jinja
relative_path(path_of(global_root.shared).value, context)
```

Target and base must belong to the same root space.

### `at(path, anchor=context)`

Strictly resolve a path and return either a node or a scalar field:

```jinja
at(path._.settings)
at("global_root.config.port")
```

### `get(path, default=none, anchor=context)`

Resolve a path with a fallback for a missing or invalid path:

```jinja
get(path.optional.timeout, 30)
```

### Node indexing

A node may be indexed directly with a `PathRef`:

```jinja
context[relative_path]
global_root[absolute_path]
```

When the path is relative, `node[path]` reanchors it to that node.

### Source helpers

```jinja
root_of(node)
source_file(node)
```

These correspond to `node.root` and `node.file`.

## Imports

These names are globals and filters:

```jinja
import_yaml("file.yaml")
import("file.yaml")
import_json("file.json")
```

For file-backed resolution, relative imports start beside the importing file.
In-memory resolution uses `base_dir`, or the current working directory when it
is omitted. Imported trees remain lazy and keep their own `root`,
`source_path`, and `file` metadata.

`import_roots` restricts resolved filesystem roots:

```python
result = resolve_file(
    "project/config.yaml",
    import_roots=["project"],
)
```

Paths are resolved before enforcement, so `..` and symlinks cannot escape an
allowed root. `None` permits any path readable by the process; `[]` denies all
imports. There is currently no CLI `import_roots` option.

An import already active in the current ancestry resolves to `null`. YAML
imports require PyYAML; JSON imports do not.

## Python API

```python
from jinest import Resolver, resolve, resolve_file, resolve_text
```

### Eager convenience

```python
result = resolve({"x": 2, "y$": "x + 1"})
```

Scalar roots (`None`, booleans, finite numbers, strings) are returned unchanged.
Python `date`, `datetime`, `time`, `bytes`, and `bytearray` are also accepted.
Unsupported scalars raise `JinestError`; with `strict=False` they become
`None`.

Undefined native/script expressions become `None` in non-strict mode, while
undefined text becomes an empty string. A script without `return` produces
`None` in both modes.

### Lazy access

```python
resolver = Resolver({
    "answer$": "6 * 7",
    "broken@": "{{ missing.value }}",
})

assert resolver.root.answer == 42
assert str(resolver.root.path) == "global_root"
result = resolver.resolve()
```

Fields evaluate once per binding. A failed evaluation clears its partial
value/child/key cache, so later access retries instead of observing incomplete
state.

### Files and text

```python
rendered = resolve_text(
    '{"x": 2, "y$": "x + 1"}',
    format="json",
    output_format="json",
)

rendered = resolve_file(
    "config.yaml",
    output="resolved.json",
    output_format="json",
)
```

JSON serialization converts date/time values to ISO 8601 strings. `bytes` and
`bytearray` become lossless Latin-1 strings: byte `0xNN` maps to Unicode
`U+00NN`. A receiver that knows a field is binary restores its exact value with
`.encode("latin-1")` after JSON decoding. JSON does not retain the distinction
between a binary field and an ordinary Latin-1 string, so that type belongs in
the surrounding schema. Mapping keys follow the same conversion rules;
unsupported keys or collisions after conversion raise `JinestError`.

### Resolver options

Convenience functions forward resolver options where applicable.

| Option | Meaning |
|---|---|
| `strict=True` | Raise for undefined/unsupported values; non-strict uses `None` or empty text |
| `in_place=False` | Write a mutable materialized root back into the original mapping/list |
| `sandboxed=True` | Use Jinja sandboxed environments |
| `globals`, `filters` | Add trusted application values/callables |
| `source_path` | Source filename for metadata and relative imports |
| `base_dir` | Import base when no source file determines it |
| `import_roots` | Allowed resolved import directories; `None` allows all, `[]` denies all |
| `function_max_depth=100` | Recursive scalar-function limit |
| `emit_messages=True` | Print collected diagnostics |
| `treat_warnings_as_errors=False` | Raise after resolution when warnings exist |
| `debug=False` | Add `at`/`in` lines to stderr diagnostics |

`Resolver.messages` remains available regardless of `emit_messages`.

## CLI

The installed command and direct module invocation are equivalent:

```bash
jinest config.yaml
python jinest.py config.yaml
jinest config.yaml -o resolved.yaml
jinest config.yaml --output-format json
jinest config.yaml --debug
jinest config.yaml --no-messages
jinest config.yaml --treat-warnings-as-errors
jinest --self-test
```

Disable the sandbox only for fully trusted templates:

```bash
jinest config.yaml --unsafe
```

## Testing

```bash
python jinest.test.py
```

Portable CLI fixtures live in `tests/<test-name>/`. Each directory contains an
`input.yml` and the byte-for-byte expected `output.yml`, so another Jinest
implementation can run the same scenarios without importing Python tests:

```bash
./run_tests.sh
./run_tests.sh 'python3 jinest.py'
JINEST_CMD='python3 jinest.py' ./run_tests.sh
```

`input.yml` is required. `output.yml`, `stderr.txt`, and `exit_code` are
optional: omitted stream files mean an empty stream, and an omitted exit code
means success (`0`). This makes the same runner suitable for negative tests.

To test another file:

```bash
JINEST_MODULE=/path/to/jinest.py python jinest.test.py
```

## Examples

The [`examples/`](examples/) directory contains runnable, commented examples
for field modes, evaluator pipelines, scripts, arrays, layers, prototypes,
paths, imports, cycles, functions, composition, self declarations, Python API
extensions, and extended scalar values.

Validate every documented result with:

```bash
python examples/validate.py
```

The suite contains 98 Python regression tests and 28 implementation-neutral
portable CLI fixtures.

## Security

Jinest uses sandboxed Jinja environments by default, but the Jinja sandbox is
not a complete security boundary for hostile templates.

Imports grant filesystem reads unless `import_roots` restricts them.
User-provided `globals` and `filters` are trusted application capabilities and
can expand what templates observe or invoke. Generated mappings/lists are
parsed as Jinest subtrees, so untrusted generated keys and inline strings are
code unless escaped.

Use OS-level isolation and minimal filesystem permissions for adversarial
input. `--unsafe` is only for fully trusted templates.

## License

Jinest is released under the [MIT License](LICENSE).

Copyright © 2026 Fedir Khodchenko.

---

<div align="center">

**Jinest — keep configuration structured until the last responsible moment.**

</div>
