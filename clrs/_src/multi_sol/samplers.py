"""Sampler classes for multi-solution algorithms."""

from typing import Tuple

from clrs._src.multi_sol.data import samplers as multisol_data_samplers
from clrs._src.samplers import Sampler


class DfsMultiSampler(Sampler):
  """DFS sampler that passes seed for multi-solution generation."""

  def _sample_data(
      self,
      length: int,
      p: Tuple[float, ...] = (0.5,),
  ):
    return multisol_data_samplers.sample_data_dfs_multi(
        self, length=length, p=p)


class BfsMultiSampler(Sampler):
  """BFS sampler that passes seed for multi-solution generation."""

  def _sample_data(
      self,
      length: int,
      p: Tuple[float, ...] = (0.5,),
  ):
    return multisol_data_samplers.sample_data_bfs_multi(
        self, length=length, p=p)


class BellmanFordMultiSampler(Sampler):
  """Bellman-Ford sampler with per-instance seed for multi-solution labels."""

  def _sample_data(
      self,
      length: int,
      p: Tuple[float, ...] = (0.5,),
      low: int = 1,
      high: int = 3,
  ):
    return multisol_data_samplers.sample_data_bellman_ford_multi(
        self, length=length, p=p, low=low, high=high)


__all__ = (
    "DfsMultiSampler",
    "BfsMultiSampler",
    "BellmanFordMultiSampler",
)
