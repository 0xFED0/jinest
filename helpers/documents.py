"""Wheel import wrapper for ``jinest.documents``."""

from jinest import documents as _namespace

load_json = _namespace.load_json
load_yaml = _namespace.load_yaml
import_json = _namespace.import_json
import_yaml = _namespace.import_yaml
import_tree = _namespace.import_tree
export_json = _namespace.export_json
export_yaml = _namespace.export_yaml

__all__ = ('load_json', 'load_yaml', 'import_json', 'import_yaml', 'import_tree', 'export_json', 'export_yaml')
