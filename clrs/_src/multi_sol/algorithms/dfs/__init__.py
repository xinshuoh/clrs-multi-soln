"""DFS multi-solution package."""

from clrs._src.multi_sol.algorithms.dfs.generator import dfs_multi
from clrs._src.multi_sol.algorithms.dfs.plugin import evaluate_dfs_multisol_batch

__all__ = ("dfs_multi", "evaluate_dfs_multisol_batch")
