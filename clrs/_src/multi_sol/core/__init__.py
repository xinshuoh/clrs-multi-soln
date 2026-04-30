"""Core interfaces and registries for multi-solution workflows."""

from clrs._src.multi_sol.core.definitions import ExtractionMethod
from clrs._src.multi_sol.core.definitions import GeneratorSamplingSource
from clrs._src.multi_sol.core.definitions import MultiSolAlgorithm
from clrs._src.multi_sol.core.definitions import MultiSolSolutionSpace
from clrs._src.multi_sol.core.definitions import RandomizedAlgorithm
from clrs._src.multi_sol.core.definitions import TrainingDistribution
from clrs._src.multi_sol.core.registry import build_overlay_specs
from clrs._src.multi_sol.core.registry import get_extension
from clrs._src.multi_sol.core.registry import get_extension_algorithms
from clrs._src.multi_sol.core.registry import list_extensions
from clrs._src.multi_sol.core.registry import register_extension
from clrs._src.multi_sol.core.registry import resolve_specs

__all__ = (
    "ExtractionMethod",
    "GeneratorSamplingSource",
    "MultiSolAlgorithm",
    "MultiSolSolutionSpace",
    "RandomizedAlgorithm",
    "TrainingDistribution",
    "build_overlay_specs",
    "get_extension",
    "get_extension_algorithms",
    "list_extensions",
    "register_extension",
    "resolve_specs",
)
