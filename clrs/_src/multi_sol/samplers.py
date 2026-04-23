"""Sampler classes for multi-solution algorithms."""

from typing import Tuple

from clrs._src.samplers import Sampler


class DfsMultiSampler(Sampler):
  """DFS sampler that passes seed for multi-solution generation."""

  def _sample_data(
      self,
      length: int,
      p: Tuple[float, ...] = (0.5,),
  ):
    graph = self._random_er_graph(
        nb_nodes=length,
        p=self._rng.choice(p),
        directed=True,
        acyclic=False,
        weighted=False)
    sub_seed = int(self._rng.randint(0, 2**31))
    return [graph, sub_seed]


class BfsMultiSampler(Sampler):
  """BFS sampler that passes seed for multi-solution generation."""

  def _sample_data(
      self,
      length: int,
      p: Tuple[float, ...] = (0.5,),
  ):
    graph = self._random_er_graph(
        nb_nodes=length,
        p=self._rng.choice(p),
        directed=False,
        acyclic=False,
        weighted=False)
    source_node = int(self._rng.choice(length))
    sub_seed = int(self._rng.randint(0, 2**31))
    return [graph, source_node, sub_seed]


class BellmanFordMultiSampler(Sampler):
  """Bellman-Ford sampler with per-instance seed for multi-solution labels."""

  def _sample_data(
      self,
      length: int,
      p: Tuple[float, ...] = (0.5,),
      low: int = 1,
      high: int = 3,
  ):
    graph = self._few_weights_random_er_graph(
        nb_nodes=length,
        p=self._rng.choice(p),
        directed=False,
        acyclic=False,
        weighted=True,
        low=low,
        high=high)
    source_node = int(self._rng.choice(length))
    sub_seed = int(self._rng.randint(0, 2**31))
    return [graph, source_node, sub_seed]


__all__ = (
    "DfsMultiSampler",
    "BfsMultiSampler",
    "BellmanFordMultiSampler",
)
