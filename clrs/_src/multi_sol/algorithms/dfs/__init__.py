"""DFS multi-solution package."""

__all__ = ("dfs_multi", "evaluate_dfs_multisol_batch")


def __getattr__(name):
  if name == "dfs_multi":
    from clrs._src.multi_sol.algorithms.dfs.generator import dfs_multi
    return dfs_multi
  if name == "evaluate_dfs_multisol_batch":
    from clrs._src.multi_sol.algorithms.dfs.plugin import (
        evaluate_dfs_multisol_batch,
    )
    return evaluate_dfs_multisol_batch
  raise AttributeError(name)
