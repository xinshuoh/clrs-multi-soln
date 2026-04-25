"""Compatibility re-exports for split multi-solution graph generators."""

from clrs._src.multi_sol.algorithms.bellman_ford.generator import bellman_ford_multi
from clrs._src.multi_sol.algorithms.bfs.generator import bfs_multi
from clrs._src.multi_sol.algorithms.dfs.generator import dfs_multi
from clrs._src.multi_sol.algorithms.mst_prim.generator import mst_prim_multi

__all__ = ("dfs_multi", "bfs_multi", "bellman_ford_multi", "mst_prim_multi")

