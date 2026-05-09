"""DFS multi-solution package."""

__all__ = ("dfs_multi")


def __getattr__(name):
  if name == "dfs_multi":
    from clrs._src.multi_sol.algorithms.dfs.generator import dfs_multi
    return dfs_multi
  raise AttributeError(name)
