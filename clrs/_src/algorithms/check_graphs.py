"""Compatibility wrappers for multi-solution graph validators."""

from clrs._src.multi_sol.validation.check_graphs import bellman_ford_cost
from clrs._src.multi_sol.validation.check_graphs import check_valid_BFpaths
from clrs._src.multi_sol.validation.check_graphs import check_valid_bfsTree
from clrs._src.multi_sol.validation.check_graphs import check_valid_dfsTree
from clrs._src.multi_sol.validation.check_graphs import check_valid_dfsTree_new

__all__ = (
    "bellman_ford_cost",
    "check_valid_BFpaths",
    "check_valid_bfsTree",
    "check_valid_dfsTree",
    "check_valid_dfsTree_new",
)

