"""Wheel import wrapper for ``jinest.runtime``."""

from jinest import runtime as _namespace

node = _namespace.node
resolve = _namespace.resolve
eval = _namespace.eval
render = _namespace.render
script = _namespace.script
literal = _namespace.literal

__all__ = ('node', 'resolve', 'eval', 'render', 'script', 'literal')
