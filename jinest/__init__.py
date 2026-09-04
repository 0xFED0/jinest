"""Wheel package adapter for the self-contained :mod:`jinest.py` runtime.

The implementation intentionally remains in the top-level single-file module;
this package only makes namespace imports available when installed as a wheel.
"""
from pathlib import Path as _Path

# In a source checkout wrappers live in the repository-level ``helpers/``
# directory. The wheel maps that directory to ``jinest.helpers`` below.
# Include the repository root while importing from an uninstalled checkout.
if str(_Path(__file__).resolve().parent.parent) not in __path__:
    __path__.append(str(_Path(__file__).resolve().parent.parent))

_runtime = _Path(__file__).resolve().parent.parent / "jinest.py"
exec(compile(_runtime.read_bytes(), str(_runtime), "exec"), globals(), globals())
del _Path, _runtime
