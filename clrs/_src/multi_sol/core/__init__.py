"""Core interfaces and registries for multi-solution workflows."""

from clrs._src.multi_sol.core.definitions import MultiSolAlgorithmDefinition
from clrs._src.multi_sol.core.definitions import MultiSolAlgorithmExtension
from clrs._src.multi_sol.core.definitions import MultiSolEvaluationDefinition
from clrs._src.multi_sol.core.definitions import MultiSolExtensionDefinition
from clrs._src.multi_sol.core.definitions import MultiSolSolutionSpace
from clrs._src.multi_sol.core.registry import build_overlay_specs
from clrs._src.multi_sol.core.registry import get_extension
from clrs._src.multi_sol.core.registry import get_extension_algorithms
from clrs._src.multi_sol.core.registry import list_extensions
from clrs._src.multi_sol.core.registry import register_extension
from clrs._src.multi_sol.core.registry import resolve_specs

__all__ = (
    "MultiSolAlgorithmDefinition",
    "MultiSolAlgorithmExtension",
    "MultiSolEvaluationDefinition",
    "MultiSolExtensionDefinition",
    "MultiSolSolutionSpace",
    "build_overlay_specs",
    "get_extension",
    "get_extension_algorithms",
    "list_extensions",
    "register_extension",
    "resolve_specs",
)
