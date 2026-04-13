"""Catalog of built-in multi-solution extension definitions."""

from clrs._src.multi_sol.algorithms.bellman_ford.definition import (
    EXTENSION as BELLMAN_FORD_MULTI_EXTENSION)
from clrs._src.multi_sol.algorithms.bfs.definition import (
    EXTENSION as BFS_MULTI_EXTENSION)
from clrs._src.multi_sol.algorithms.dfs.definition import (
    EXTENSION as DFS_MULTI_EXTENSION)

BUILTIN_EXTENSIONS = (
    DFS_MULTI_EXTENSION,
    BFS_MULTI_EXTENSION,
    BELLMAN_FORD_MULTI_EXTENSION,
)

__all__ = (
    "BELLMAN_FORD_MULTI_EXTENSION",
    "BFS_MULTI_EXTENSION",
    "BUILTIN_EXTENSIONS",
    "DFS_MULTI_EXTENSION",
)
