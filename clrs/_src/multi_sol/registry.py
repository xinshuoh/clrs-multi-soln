"""Registry data for multi-solution algorithms."""

from __future__ import annotations

from typing import Dict

from clrs._src.multi_sol.algorithms.bellman_ford import definition as bellman_ford
from clrs._src.multi_sol.algorithms.bfs import definition as bfs
from clrs._src.multi_sol.algorithms.dfs import definition as dfs
from clrs._src.multi_sol.interfaces import MultiSolAlgorithm
from clrs._src.multi_sol.algorithms.mst_prim import definition as mst_prim

MULTI_SOL_ALGS: Dict[str, MultiSolAlgorithm] = {
    algorithm.name: algorithm for algorithm in (
        dfs.ALGORITHM,
        bfs.ALGORITHM,
        bellman_ford.ALGORITHM,
        mst_prim.ALGORITHM,
    )
}
__all__ = ("MULTI_SOL_ALGS",)
