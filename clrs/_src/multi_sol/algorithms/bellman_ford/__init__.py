"""Bellman-Ford multi-solution package."""

__all__ = ("bellman_ford_multi", "evaluate_bf_multisol_batch")


def __getattr__(name):
  if name == "bellman_ford_multi":
    from clrs._src.multi_sol.algorithms.bellman_ford.generator import (
        bellman_ford_multi,
    )
    return bellman_ford_multi
  if name == "evaluate_bf_multisol_batch":
    from clrs._src.multi_sol.algorithms.bellman_ford.plugin import (
        evaluate_bf_multisol_batch,
    )
    return evaluate_bf_multisol_batch
  raise AttributeError(name)
