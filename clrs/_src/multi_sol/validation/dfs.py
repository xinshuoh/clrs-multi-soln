"""DFS validators."""

import copy
import graphlib as gl

import networkx as nx
import numpy as np

from clrs._src.algorithms import dfs_verification_tester


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
  """Check whether self-loops and parent choices are DFS-order-consistent."""
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


def check_valid_dfs_tree(adjacency, parent_tree, _source_unused):
  parent_tree = np.asarray(parent_tree).astype(int)
  return dfs_verification_tester.dfsverify(adjacency, parent_tree)


__all__ = (
    "replace_self_loops_with_minus1",
    "are_valid_edges_parents",
    "are_valid_order_parents",
    "is_acyclic",
    "check_valid_dfsTree",
    "check_valid_dfsTree_new",
    "check_valid_dfs_tree",
)
