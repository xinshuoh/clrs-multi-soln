"""Compatibility wrappers for multi-solution graph validators."""

from clrs._src.multi_sol.algorithms.bellman_ford.validator import bellman_ford_cost
from clrs._src.multi_sol.algorithms.bellman_ford.validator import check_valid_BFpaths
from clrs._src.multi_sol.algorithms.bfs.validator import check_valid_bfsTree
from clrs._src.multi_sol.algorithms.dfs.validator import check_valid_dfsTree
from clrs._src.multi_sol.algorithms.dfs.validator import check_valid_dfsTree_new
from clrs._src.multi_sol.algorithms.mst_prim.validator import check_valid_mstPrimTree
from clrs._src.multi_sol.algorithms.mst_prim.validator import is_source_rooted_tree
from clrs._src.multi_sol.algorithms.mst_prim.validator import prim_mst_weight

__all__ = (
    "bellman_ford_cost",
    "check_valid_BFpaths",
    "check_valid_bfsTree",
    "check_valid_dfsTree",
    "check_valid_dfsTree_new",
    "check_valid_mstPrimTree",
    "is_source_rooted_tree",
    "prim_mst_weight",
)
