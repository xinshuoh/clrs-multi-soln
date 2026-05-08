"""Compatibility wrappers for BFS multi-solution sampling methods.

This module keeps the legacy import path while delegating implementation to the
new modular framework under `clrs._src.multi_sol`.
"""

from clrs._src.multi_sol.algorithms.bfs.extractors import _bfs_beam_sampler as bfs_beam_sampler
from clrs._src.multi_sol.algorithms.bfs.extractors import extract_prim as prim_like_sampler
from clrs._src.multi_sol.algorithms.bfs.extractors import extract_beam as sample_bfs_beam
from clrs._src.multi_sol.algorithms.bfs.extractors import extract_categorical as sample_bfs_categorical
from clrs._src.multi_sol.algorithms.bfs.extractors import extract_prim as sample_bfs_prim

__all__ = (
    "sample_bfs_prim",
    "prim_like_sampler",
    "sample_bfs_categorical",
    "sample_bfs_beam",
    "bfs_beam_sampler",
)

