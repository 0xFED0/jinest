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

> **Status:** Jinest `0.7.0` is a single-file prototype with a standalone regression suite. The public API may still evolve before `1.0`.

## Highlights

- Lazy field resolution and dependency tracking.
- Native Jinja expressions with `$`.
- Full text templates with `@`.
- Multiline Jinja scripts with `^`, `%` line statements, and native `return`.
- Lazy default and override layers without eager dictionary merging.
- Destination-aware reusable prototypes.
- Independent source roots for imported YAML and JSON trees.
- Structured `PathRef` objects and explicit navigation helpers.
- Source and destination metadata on every mapping/list node.
- Cycle handling consistent across fields, layers, and imports.

## Installation

Jinest currently ships as one Python file. Copy `jinest.py` into your project, then install Jinja:

```bash
python -m pip install "jinja2>=3.1"
```

YAML support additionally requires PyYAML:

```bash
python -m pip install pyyaml
```

## Syntax

| Form | Meaning |
|---|---|
| `name@` | Full Jinja text template; result is a string |
| `name$` | One Jinja expression; result preserves its native type |
| `name^` | Multiline Jinja script; `return` produces a native value |
| `<<$`, `<<N$` | Native-expression default layer |
| `<<^`, `<<N^` | Script default layer |
| `<<!$`, `<<N!$` | Native-expression override layer |
| `<<!^`, `<<N!^` | Script override layer |

Local field priority is:

```text
name > name^ > name$ > name@
```

Ignored lower-priority alternatives are not parsed or evaluated.

## Native expressions: `$`

A `$` field contains exactly one Jinja expression, without `{{ ... }}`:

```yaml
values:
  x: 2
  y: 3
  sum$: "x + y"
  enabled$: "sum > 4"
  object$: "{'x': x, 'y': y}"
```

## Text templates: `@`

An `@` field is a complete Jinja template:

```yaml
release:
  product: Jinest
  version: 0.7.0
  label@: "{{ product }} v{{ version }}"
```

## Multiline scripts: `^`

A `^` field uses Jinja line statements prefixed with `%` and may return any native value:

```yaml
result^: |
  % set sum = x + y
  % set doubled = sum * 2

  % if doubled > 10
    % return {
      "value": doubled,
      "large": true
    }
  % endif

  % return {
    "value": doubled,
    "large": false
  }
```

Rules:

- `return expression` immediately terminates the script.
- `return` without an expression returns `null`.
- Reaching the end without `return` also returns `null`.
- `return` works inside `if`, `for`, and nested Jinja blocks.
- Standard `{% ... %}` syntax remains accepted, but `%` line statements are the intended notation.

## Lazy layers

Mappings may inherit lazy default and override layers:

```yaml
defaults:
  host: localhost
  port: 8000

overrides:
  port: 443
  secure: true

service:
  <<^: |
    % return root.defaults

  port: 8080

  <<!^: |
    % if force_secure
      % return root.overrides
    % endif
    % return null
```

Precedence is:

```text
defaults → local fields → overrides
```

Lookup runs in reverse:

```text
last override → first override → local → last default → first default
```

Default and override numbering are independent. `$` and `^` affect only how a layer source is computed, not its precedence.

A `null` merge result is an empty layer. Any other result must be a mapping.

## Evaluation context

Every template or expression receives:

| Name | Meaning |
|---|---|
| `context` | Destination mapping/list currently being evaluated |
| `_` | Parent of `context` |
| `path` | Destination `PathRef` for the current context or array item |
| `origin` | Source mapping/list where the winning field or layer was declared |
| `root` | Root of the source tree that owns `origin` |
| `global_root` | Top-level root of the original `Resolver` |

This split allows imported prototypes to use their own absolute source references while adapting relative references to the destination:

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

## Lazy arrays

Lists are lazy containers. Direct lists under suffixes resolve each string item independently:

```yaml
native_values$:
  - x + 1
  - root.constant
  - 123

text_values@:
  - "Value: {{ x }}"
  - "Path: {{ path }}"
  - 123

script_values^:
  - |
      % set value = x * 2
      % return value
  - |
      % return path
  - 123
```

Non-string items remain literal. A list returned by a string expression or script is treated as already computed and is not executed a second time.

Array indices participate in paths:

```text
global_root.items[5].object
```

## Imports

The following are available as globals and filters:

```jinja
import_yaml("file.yaml")
import("file.yaml")
import_json("file.json")
```

Relative paths are resolved from the file containing the import. Imported trees remain lazy and retain their own `root`, `source_path`, and `file` metadata.

An import already active in the current import ancestry resolves to `null`, matching ordinary field-cycle semantics.

## Python API

```python
from jinest import Resolver, resolve, resolve_file, resolve_text
```

### Eager convenience

```python
result = resolve({
    "x": 2,
    "y$": "x + 1",
})
```

Scalar roots such as `null`, booleans, numbers, and strings are returned unchanged.
Python `date`, `datetime`, `time`, `bytes`, and `bytearray` values are also accepted.
Unsupported scalar types raise `JinestError`; with `strict=False` they resolve to
`None`. Missing native expressions and script returns likewise become `None` in
non-strict mode, while missing text values render as an empty string.

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

### Files

```python
text = resolve_file(
    "config.yaml",
    output="resolved.json",
    output_format="json",
)
```

When serializing JSON, date/time values use ISO 8601 strings. Every byte in a
`bytes` or `bytearray` value is emitted as one pure Unicode escape (`\uHHHH`).

## CLI

```bash
python jinest.py config.yaml
python jinest.py config.yaml -o resolved.yaml
python jinest.py config.yaml --output-format json
python jinest.py --self-test
```

Disable the Jinja sandbox only for fully trusted input:

```bash
python jinest.py config.yaml --unsafe
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
covering field modes, scripts, arrays, layers, prototypes, paths, imports,
cycles, Python API extensions, extended scalar values, and scalar roots.

Validate every documented result with:

```bash
python examples/validate.py
```

The test suite contains 59 regression tests covering expressions, templates, scripts, layer precedence, prototypes, arrays, imports, cycles, metadata, path parsing, navigation, and YAML syntax.

## Security

Jinest uses a sandboxed Jinja environment by default. The sandbox reduces exposure but is not a complete security boundary for hostile templates. Imported files also grant filesystem reads available to the running process. Use an OS-level sandbox and restrict accessible directories for adversarial inputs.

## License

Jinest is released under the [MIT License](LICENSE).

Copyright © 2026 Fedir Khodchenko.

---

<div align="center">

**Jinest — keep configuration structured until the last responsible moment.**

</div>
