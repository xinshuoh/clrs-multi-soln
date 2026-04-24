"""BFS multi-solution package."""

from clrs._src.multi_sol.algorithms.bfs.generator import bfs_multi
from clrs._src.multi_sol.algorithms.bfs.plugin import evaluate_bfs_multisol_batch

__all__ = ("bfs_multi", "evaluate_bfs_multisol_batch")
