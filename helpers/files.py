"""Wheel import wrapper for ``jinest.files``."""

from jinest import files as _namespace

file_path = _namespace.file_path
read_text = _namespace.read_text
read_lines = _namespace.read_lines
read_bytes = _namespace.read_bytes
file_exists = _namespace.file_exists

__all__ = ('file_path', 'read_text', 'read_lines', 'read_bytes', 'file_exists')
