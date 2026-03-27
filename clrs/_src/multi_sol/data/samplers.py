"""Helper routines for multi-solution sampler payload generation."""

from typing import Tuple


def sample_data_dfs_multi(sampler, length: int, p: Tuple[float, ...] = (0.5,)):
  """Return DFS-multi sampler payload [graph, seed]."""
  graph = sampler._random_er_graph(  # pylint: disable=protected-access
      nb_nodes=length,
      p=sampler._rng.choice(p),  # pylint: disable=protected-access
      directed=True,
      acyclic=False,
      weighted=False)
  sub_seed = int(sampler._rng.randint(0, 2**31))  # pylint: disable=protected-access
  return [graph, sub_seed]


def sample_data_bfs_multi(sampler, length: int, p: Tuple[float, ...] = (0.5,)):
  """Return BFS-multi sampler payload [graph, source, seed]."""
  graph = sampler._random_er_graph(  # pylint: disable=protected-access
      nb_nodes=length,
      p=sampler._rng.choice(p),  # pylint: disable=protected-access
      directed=False,
      acyclic=False,
      weighted=False)
  source_node = int(sampler._rng.choice(length))  # pylint: disable=protected-access
  sub_seed = int(sampler._rng.randint(0, 2**31))  # pylint: disable=protected-access
  return [graph, source_node, sub_seed]


def sample_data_bellman_ford_multi(
    sampler,
    length: int,
    p: Tuple[float, ...] = (0.5,),
    low: int = 1,
    high: int = 3,
):
  """Return BF-multi sampler payload [graph, source, seed]."""
  graph = sampler._few_weights_random_er_graph(  # pylint: disable=protected-access
      nb_nodes=length,
      p=sampler._rng.choice(p),  # pylint: disable=protected-access
      directed=False,
      acyclic=False,
      weighted=True,
      low=low,
      high=high)
  source_node = int(sampler._rng.choice(length))  # pylint: disable=protected-access
  sub_seed = int(sampler._rng.randint(0, 2**31))  # pylint: disable=protected-access
  return [graph, source_node, sub_seed]


__all__ = (
    "sample_data_dfs_multi",
    "sample_data_bfs_multi",
    "sample_data_bellman_ford_multi",
)
