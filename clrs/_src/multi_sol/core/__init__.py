"""Core interfaces and registries for multi-solution workflows."""

from clrs._src.multi_sol.core.registry import MultiSolAlgorithmExtension
from clrs._src.multi_sol.core.registry import MultiSolExtensionDefinition
from clrs._src.multi_sol.core.registry import build_overlay_specs
from clrs._src.multi_sol.core.registry import get_extension
from clrs._src.multi_sol.core.registry import get_extension_algorithms
from clrs._src.multi_sol.core.registry import list_extensions
from clrs._src.multi_sol.core.registry import register_extension
from clrs._src.multi_sol.core.registry import resolve_specs

__all__ = (
    "MultiSolAlgorithmExtension",
    "MultiSolExtensionDefinition",
    "build_overlay_specs",
    "get_extension",
    "get_extension_algorithms",
    "list_extensions",
    "register_extension",
    "resolve_specs",
)
