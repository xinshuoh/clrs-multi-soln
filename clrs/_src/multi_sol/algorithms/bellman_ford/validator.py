"""Bellman-Ford validators."""

import chex
import numpy as np


def bellman_ford_cost(adjacency, source):
  """Compute Bellman-Ford shortest path costs from source."""
  chex.assert_rank(adjacency, 2)
  d = np.zeros(adjacency.shape[0])
  msk = np.zeros(adjacency.shape[0])
  d[source] = 0
  msk[source] = 1
  while True:
    prev_d = np.copy(d)
    prev_msk = np.copy(msk)
    for u in range(adjacency.shape[0]):
      for v in range(adjacency.shape[0]):
        if prev_msk[u] == 1 and adjacency[u, v] != 0:
          if msk[v] == 0 or prev_d[u] + adjacency[u, v] < d[v]:
            d[v] = prev_d[u] + adjacency[u, v]
          msk[v] = 1
    if np.all(d == prev_d):
      break
  return d


def check_valid_BFpaths(adjacency, source, parentpath):
  """Validate Bellman-Ford parent pointers by shortest-path cost equivalence."""
  true_costs = bellman_ford_cost(adjacency, source)
  parentpath = np.array(parentpath).astype(int)

  bf_tree_adj = np.zeros((len(parentpath), len(parentpath)))
  for i in range(len(parentpath)):
    if adjacency[parentpath[i], i] == 0 and parentpath[i] != i:
      return False
    bf_tree_adj[parentpath[i], i] = adjacency[parentpath[i], i]

  model_costs = bellman_ford_cost(bf_tree_adj, source)
  return (true_costs == model_costs).all()


def check_valid_bf_paths(adjacency, parent_tree, source):
  return check_valid_BFpaths(adjacency, source, parent_tree)


__all__ = (
    "bellman_ford_cost",
    "check_valid_BFpaths",
    "check_valid_bf_paths",
)
