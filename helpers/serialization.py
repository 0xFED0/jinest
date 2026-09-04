"""Wheel import wrapper for ``jinest.serialization``."""

from jinest import serialization as _namespace

from_json = _namespace.from_json
from_yaml = _namespace.from_yaml
to_json = _namespace.to_json
to_yaml = _namespace.to_yaml
json_normalize = _namespace.json_normalize
yaml_normalize = _namespace.yaml_normalize
serialize = _namespace.serialize

__all__ = ('from_json', 'from_yaml', 'to_json', 'to_yaml', 'json_normalize', 'yaml_normalize', 'serialize')
