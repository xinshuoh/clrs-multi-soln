"""Modular multi-solution infrastructure for CLRS-based NAR experiments."""

from clrs._src.multi_sol.algorithms.bellman_ford import evaluate_bf_multisol_batch
from clrs._src.multi_sol.algorithms.bfs.plugin import evaluate_bfs_multisol_batch
from clrs._src.multi_sol.algorithms.dfs import evaluate_dfs_multisol_batch

__all__ = (
    "evaluate_bf_multisol_batch",
    "evaluate_bfs_multisol_batch",
    "evaluate_dfs_multisol_batch",
)
