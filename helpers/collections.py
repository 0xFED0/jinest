"""Wheel import wrapper for ``jinest.collections``."""

from jinest import collections as _namespace

combine = _namespace.combine
union = _namespace.union
intersect = _namespace.intersect
difference = _namespace.difference
symmetric_difference = _namespace.symmetric_difference

__all__ = ('combine', 'union', 'intersect', 'difference', 'symmetric_difference')
