"""Central registry of built-in multi-solution algorithms."""

from clrs._src.multi_sol.algorithms.bellman_ford import definition as bellman_ford
from clrs._src.multi_sol.algorithms.bfs import definition as bfs
from clrs._src.multi_sol.algorithms.dfs import definition as dfs
from clrs._src.multi_sol.algorithms.mst_prim import definition as mst_prim


MULTI_SOL_ALGS = {
    dfs.DEFINITION.algorithm_name: dfs.DEFINITION,
    bfs.DEFINITION.algorithm_name: bfs.DEFINITION,
    bellman_ford.DEFINITION.algorithm_name: bellman_ford.DEFINITION,
    mst_prim.DEFINITION.algorithm_name: mst_prim.DEFINITION,
}

BUILTIN_DEFINITIONS = tuple(MULTI_SOL_ALGS.values())

__all__ = ("MULTI_SOL_ALGS", "BUILTIN_DEFINITIONS")

