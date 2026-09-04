"""Wheel import wrapper for ``jinest.path``."""

from jinest import path as _namespace

normalize_path = _namespace.normalize_path
absolute_path = _namespace.absolute_path
relative_path = _namespace.relative_path
path_of = _namespace.path_of
source_path_of = _namespace.source_path_of
at = _namespace.at
get = _namespace.get
root_of = _namespace.root_of
source_file = _namespace.source_file
source_dir = _namespace.source_dir

__all__ = ('normalize_path', 'absolute_path', 'relative_path', 'path_of', 'source_path_of', 'at', 'get', 'root_of', 'source_file', 'source_dir')
