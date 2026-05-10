"""Compatibility forwarding for multi-solution registry imports."""

from clrs._src.multi_sol.registry import get
from clrs._src.multi_sol.registry import names
from clrs._src.multi_sol.registry import register_extension

__all__ = (
    "get",
    "names",
    "register_extension",
)
