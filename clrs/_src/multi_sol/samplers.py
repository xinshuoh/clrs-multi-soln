"""Compatibility aliases for multi-solution samplers.

Multi-solution sampler classes are now defined in `clrs._src.samplers`.
"""

from clrs._src import samplers as base_samplers


class DfsMultiSampler(base_samplers.DfsMultiSampler):
  pass


class BfsMultiSampler(base_samplers.BfsMultiSampler):
  pass


class BellmanFordMultiSampler(base_samplers.BellmanFordMultiSampler):
  pass


class MSTPrimSampler(base_samplers.MSTPrimSampler):
  pass


class MSTPrimMultiSampler(base_samplers.MSTPrimMultiSampler):
  pass

__all__ = (
    "DfsMultiSampler",
    "BfsMultiSampler",
    "BellmanFordMultiSampler",
    "MSTPrimSampler",
    "MSTPrimMultiSampler",
)
