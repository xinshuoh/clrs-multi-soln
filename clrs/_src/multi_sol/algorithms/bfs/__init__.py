"""BFS multi-solution plugin package."""

from clrs._src.multi_sol.algorithms.bfs.definition import EXTENSION
from clrs._src.multi_sol.algorithms.bfs.plugin import evaluate_bfs_multisol_batch

__all__ = ("EXTENSION", "evaluate_bfs_multisol_batch")
