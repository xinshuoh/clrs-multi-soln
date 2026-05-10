"""Core interfaces and registries for multi-solution workflows."""

from clrs._src.multi_sol.core.definitions import ExtractionMethod
from clrs._src.multi_sol.core.definitions import AlgorithmBaseline
from clrs._src.multi_sol.core.definitions import MultiSolAlgorithm
from clrs._src.multi_sol.core.definitions import MultiSolTrainingConfig
from clrs._src.multi_sol.core.definitions import MultiSolSolutionSpace
from clrs._src.multi_sol.core.definitions import RandomizedAlgorithm
from clrs._src.multi_sol.core.registry import get
from clrs._src.multi_sol.core.registry import names
from clrs._src.multi_sol.core.registry import register_extension

__all__ = (
    "AlgorithmBaseline",
    "ExtractionMethod",
    "MultiSolAlgorithm",
    "MultiSolTrainingConfig",
    "MultiSolSolutionSpace",
    "RandomizedAlgorithm",
    "get",
    "names",
    "register_extension",
)
