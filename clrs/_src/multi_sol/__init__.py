"""Modular multi-solution infrastructure for CLRS-based NAR experiments."""

__all__ = (
    "evaluate_bf_multisol_batch",
    "evaluate_bfs_multisol_batch",
    "evaluate_dfs_multisol_batch",
)


def __getattr__(name):
  if name == "evaluate_bf_multisol_batch":
    from clrs._src.multi_sol.algorithms.bellman_ford.plugin import (
        evaluate_bf_multisol_batch,
    )
    return evaluate_bf_multisol_batch
  if name == "evaluate_bfs_multisol_batch":
    from clrs._src.multi_sol.algorithms.bfs.plugin import (
        evaluate_bfs_multisol_batch,
    )
    return evaluate_bfs_multisol_batch
  if name == "evaluate_dfs_multisol_batch":
    from clrs._src.multi_sol.algorithms.dfs.plugin import (
        evaluate_dfs_multisol_batch,
    )
    return evaluate_dfs_multisol_batch
  raise AttributeError(name)
