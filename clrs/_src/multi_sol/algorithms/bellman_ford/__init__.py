"""Bellman-Ford multi-solution package."""

__all__ = ("bellman_ford_multi")


def __getattr__(name):
  if name == "bellman_ford_multi":
    from clrs._src.multi_sol.algorithms.bellman_ford.generator import (
        bellman_ford_multi,
    )
    return bellman_ford_multi
  raise AttributeError(name)
