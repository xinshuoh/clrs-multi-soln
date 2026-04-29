"""BFS multi-solution package."""

__all__ = ("bfs_multi", "evaluate_bfs_multisol_batch")


def __getattr__(name):
  if name == "bfs_multi":
    from clrs._src.multi_sol.algorithms.bfs.generator import bfs_multi
    return bfs_multi
  if name == "evaluate_bfs_multisol_batch":
    from clrs._src.multi_sol.algorithms.bfs.plugin import (
        evaluate_bfs_multisol_batch,
    )
    return evaluate_bfs_multisol_batch
  raise AttributeError(name)
