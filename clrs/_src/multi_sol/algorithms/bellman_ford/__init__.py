"""Bellman-Ford multi-solution plugin package."""

from clrs._src.multi_sol.algorithms.bellman_ford.definition import EXTENSION
from clrs._src.multi_sol.algorithms.bellman_ford.plugin import evaluate_bf_multisol_batch

__all__ = ("EXTENSION", "evaluate_bf_multisol_batch")
