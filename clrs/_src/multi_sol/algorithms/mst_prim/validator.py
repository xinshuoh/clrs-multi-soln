"""MST-Prim validators."""

import chex
import numpy as np

from clrs._src.multi_sol.algorithms.common import validator_utils


def prim_mst_weight(adjacency, source):
  """Return Prim MST weight and the component reached from source."""
  chex.assert_rank(adjacency, 2)
  n = adjacency.shape[0]
  visited = np.zeros(n, dtype=bool)
  key = np.full(n, np.inf)
  key[source] = 0.0
  total = 0.0

  for _ in range(n):
    unvisited_keys = np.where(visited, np.inf, key)
    u = int(np.argmin(unvisited_keys))
    if np.isinf(unvisited_keys[u]):
      break
    visited[u] = True
    total += key[u]
    for v in range(n):
      if not visited[v] and adjacency[u, v] != 0 and adjacency[u, v] < key[v]:
        key[v] = adjacency[u, v]

  return total, visited


def is_source_rooted_tree(parent_tree, source, node_mask=None):
  """Check that masked nodes are reached by following children from source."""
  n = len(parent_tree)
  if node_mask is None:
    node_mask = np.ones(n, dtype=bool)
  children = [[] for _ in range(n)]
  for v, parent in enumerate(parent_tree):
    if node_mask[v] and v != source:
      children[parent].append(v)

  seen = np.zeros(n, dtype=bool)
  stack = [source]
  while stack:
    u = stack.pop()
    if seen[u]:
      return False
    seen[u] = True
    stack.extend(children[u])
  return np.array_equal(seen, node_mask)


def check_valid_mstPrimTree(adjacency, pi, s):
  """Validate MST-Prim parent pointers by MST weight equivalence."""
  if not validator_utils.is_square_adjacency(adjacency):
    return False

  n = adjacency.shape[0]
  if s < 0 or s >= n:
    return False

  pi = validator_utils.coerce_parent_array(pi, n)
  if pi is None:
    return False
  if pi[s] != s:
    return False

  true_weight, reachable = prim_mst_weight(adjacency, s)
  sampled_weight = 0.0
  for v in range(n):
    parent = int(pi[v])
    if not reachable[v]:
      if parent != v:
        return False
      continue
    if v == s:
      continue
    if parent == v:
      return False
    if not reachable[parent]:
      return False
    if adjacency[parent, v] == 0:
      return False
    sampled_weight += adjacency[parent, v]

  if not is_source_rooted_tree(pi, s, reachable):
    return False
  return bool(np.isclose(sampled_weight, true_weight))


def check_valid_mst_prim_tree(adjacency, parent_tree, source):
  return check_valid_mstPrimTree(adjacency, parent_tree, source)


__all__ = (
    "prim_mst_weight",
    "is_source_rooted_tree",
    "check_valid_mstPrimTree",
    "check_valid_mst_prim_tree",
)
