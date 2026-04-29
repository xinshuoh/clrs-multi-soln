"""Built-in multi-solution algorithm definitions."""

from clrs._src.multi_sol.algorithms.bellman_ford import definition as bellman_ford
from clrs._src.multi_sol.algorithms.bfs import definition as bfs
from clrs._src.multi_sol.algorithms.dfs import definition as dfs
from clrs._src.multi_sol.algorithms.mst_prim import definition as mst_prim


BUILTIN_DEFINITIONS = (
    dfs.DEFINITION,
    bfs.DEFINITION,
    bellman_ford.DEFINITION,
    mst_prim.DEFINITION,
)

__all__ = ("BUILTIN_DEFINITIONS",)
