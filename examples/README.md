# Jinest examples

Every numbered directory contains a commented, runnable `example.yml` and an
annotated expected `result.yml`. Commands below assume the repository root as
the current directory.

Run a regular example:

```bash
python jinest.py examples/01_field_modes/example.yml \
  --output-format yaml \
  -o /tmp/jinest-result.yml
```

`08_python_api` needs its runner because it installs custom Jinja globals and
filters:

```bash
python examples/08_python_api/run.py
```

Validate the complete collection:

```bash
python examples/validate.py
```

The validator executes every input, compares parsed YAML values with the
commented result files, and checks the exact JSON byte escapes in example 09.

## Coverage

| Directory | Features demonstrated |
|---|---|
| `01_field_modes` | Literal fields, `@`, `$`, `^`, native values, local priority, ignored alternatives |
| `02_scripts_and_arrays` | Line statements, standard Jinja blocks, native `return`, loops, early/empty/implicit return, `$`/`@`/`^` arrays, per-item paths, slices and negative indexes |
| `03_lazy_layers` | Native/script default and override layers, independent numbering, precedence, null and recursive layers |
| `04_prototypes_and_context` | Reusable destination-bound prototypes; `context`, `_`, `path`, `origin`, `root`, and `global_root` |
| `05_paths_and_metadata` | `PathRef`, non-identifier keys, metadata collisions, all path/source helpers, node indexing and rebinding |
| `06_imports` | YAML/JSON imports, `import` alias, filter forms, nested relative imports, independent source roots |
| `07_cycles` | Field, script, layer, null-layer, and cross-file import cycles |
| `08_python_api` | Custom globals/filters, lazy access, `in_place=True`, and `strict=False` |
| `09_extended_values` | Native date/datetime/bytes plus ISO 8601 and pure `\uHHHH` JSON normalization |
| `10_scalar_root` | Non-container scalar root documents |
| `11_functions` | Native/text/script and structural functions, defaults, namespaces, and call-site rebinding |
| `12_matrix_by_code` | Building a version/module matrix with script loops and structural functions |
| `13_matrix_by_composition` | The same version/module matrix with Cartesian structural and text compose declarations |

Comments are intentionally retained in checked-in `result.yml` files. A fresh
CLI output is therefore value-equivalent rather than byte-for-byte identical.
