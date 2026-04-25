"""Graph validation helpers used by multi-solution extraction workflows."""

from __future__ import annotations

import copy
import graphlib as gl

import chex
import networkx as nx
import numpy as np


def replace_self_loops_with_minus1(pi):
  for i in range(len(pi)):
    if pi[i] == i:
      pi[i] = -1
  return pi


def are_valid_edges_parents(adjacency, pi):
  pi = np.array(pi).astype(int)
  for i in range(len(pi)):
    parent = pi[i]
    if parent != i and adjacency[parent][i] == 0:
      return False
  return True


def are_valid_order_parents(adjacency, pi):
  """Checks whether self-loops and parent choices are DFS-order-consistent."""
  graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
  for i in range(len(pi)):
    if pi[i] == i:
      for j in range(i):
        if nx.has_path(graph, j, i):
          return False
    else:
      for j in range(i):
        if nx.has_path(graph, j, i):
          if not nx.has_path(graph, j, pi[i]):
            return False
          break
  return True


def is_acyclic(pi):
  """Check whether predecessor array induces an acyclic dependency graph."""
  ts = gl.TopologicalSorter()
  for i in range(len(pi)):
    ts.add(i, pi[i])
  try:
    ts.prepare()
    return True
  except ValueError as exc:
    if isinstance(exc, gl.CycleError):
      return False
    raise exc


def check_valid_dfsTree(adjacency, pi):
  """Validate whether predecessor array is a DFS tree under ordered restarts."""
  pi = copy.deepcopy(pi)
  if pi[0] != 0:
    return False
  if not are_valid_edges_parents(adjacency, pi):
    return False
  if not are_valid_order_parents(adjacency, pi):
    return False
  pi = replace_self_loops_with_minus1(pi)
  return is_acyclic(pi)


def check_valid_dfsTree_new(adjacency, pi):
  """Compatibility alias kept for existing tests and callers."""
  return check_valid_dfsTree(adjacency, pi)


def bellman_ford_cost(adjacency, source):
  """Bellman-Ford shortest path cost helper."""
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


def check_valid_bfsTree(adjacency, pi, s):
  """Validate bfs_multi-style parent choices including level tie-break consistency."""
  pi = copy.deepcopy(pi)
  assert len(pi) == adjacency.shape[0], "pi length must match number of nodes in A"
  pi = np.asarray(pi).astype(int)
  n = adjacency.shape[0]
  if np.any(pi < 0) or np.any(pi >= n):
    return False

  graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
  dist = nx.single_source_shortest_path_length(graph, s)
  if pi[s] != s:
    return False

  for i in range(n):
    if i == s:
      continue
    if i not in dist:
      if pi[i] != i:
        return False
    else:
      parent = int(pi[i])
      if parent == i:
        return False
      if adjacency[parent, i] == 0:
        return False
      if parent not in dist or dist[parent] != dist[i] - 1:
        return False

  levels = {}
  for node, lvl in dist.items():
    levels.setdefault(lvl, []).append(node)

  max_level = max(levels.keys(), default=0)
  for lvl in range(1, max_level + 1):
    prev_level = levels.get(lvl - 1, [])
    cur_level = levels.get(lvl, [])
    if not cur_level:
      continue

    ordering_constraints = nx.DiGraph()
    ordering_constraints.add_nodes_from(prev_level)
    for child in cur_level:
      parent = int(pi[child])
      candidate_parents = [u for u in prev_level if adjacency[u, child] != 0]
      if parent not in candidate_parents:
        return False
      for other in candidate_parents:
        if other != parent:
          ordering_constraints.add_edge(parent, other)

    if not nx.is_directed_acyclic_graph(ordering_constraints):
      return False

  return True


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
  chex.assert_rank(adjacency, 2)
  if adjacency.shape[0] != adjacency.shape[1]:
    return False

  n = adjacency.shape[0]
  if s < 0 or s >= n:
    return False
  if len(pi) != n:
    return False

  pi = np.asarray(pi).astype(int)
  if np.any(pi < 0) or np.any(pi >= n):
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
  return np.isclose(sampled_weight, true_weight)
