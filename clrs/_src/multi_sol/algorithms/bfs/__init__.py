"""BFS multi-solution package."""

__all__ = ("bfs_multi")


def __getattr__(name):
  if name == "bfs_multi":
    from clrs._src.multi_sol.algorithms.bfs.generator import bfs_multi
    return bfs_multi
  raise AttributeError(name)
