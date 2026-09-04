"""Wheel package adapter for the self-contained :mod:`jinest.py` runtime.

The implementation intentionally remains in the top-level single-file module;
this package only makes namespace imports available when installed as a wheel.
"""
from pathlib import Path as _Path

# In a source checkout wrappers live in the repository-level ``helpers/``
# directory. The wheel maps that directory to ``jinest.helpers`` below.
# Include the repository root while importing from an uninstalled checkout.
# ``jinest.test.py`` may also load this file under an alias to test an
# installed wheel; such an alias has no package ``__path__`` and must still
# execute the single-file runtime normally.
_package_parent = str(_Path(__file__).resolve().parent.parent)
_package_paths = globals().get("__path__")
if _package_paths is not None and _package_parent not in _package_paths:
    _package_paths.append(_package_parent)

_runtime = _Path(__file__).resolve().parent.parent / "jinest.py"
exec(compile(_runtime.read_bytes(), str(_runtime), "exec"), globals(), globals())
del _Path, _runtime, _package_parent, _package_paths
