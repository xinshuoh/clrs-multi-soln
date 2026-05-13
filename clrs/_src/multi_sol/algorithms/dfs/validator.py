"""DFS validator."""

import copy
import graphlib as gl

import networkx as nx
import numpy as np


def dfsverify(G, F):
  if isinstance(F, (list, np.ndarray)):  # check if its a parent tree format, convert
    if np.ndim(F) == 1:
      F = no_self_loops_parent_tree_to_adj_matrix(F)
  if not preprocess(F):
    return False
  if isinstance(G, np.ndarray):
    G = nx.from_numpy_array(G, create_using=nx.DiGraph)
  if isinstance(F, np.ndarray):
    F = nx.from_numpy_array(F, create_using=nx.DiGraph)
  # Check that F is a subgraph of G
  for u, v in F.edges():  # pyright: ignore[reportAttributeAccessIssue]
    if not G.has_edge(u, v):
      return False
  colors = [0] * len(G)

  for i in range(len(G)):  # outer restart loop
    if not ccv(G, F, i, colors):
      #print(f'fails on node {i}')
      return False
  return True


def dfsverify_simple(adjacency, predecessor):
  """Validate a DFS predecessor forest using the recursive subroot criterion.

  This implements the simpler recursive checker used in the accompanying
  pseudocode. It intentionally does not replace `dfsverify`; callers can swap it
  into `check_valid_dfs_tree` when they want to evaluate that criterion.
  """
  graph = (nx.from_numpy_array(adjacency, create_using=nx.DiGraph) if isinstance(
      adjacency, np.ndarray) else adjacency)
  predecessor = _as_predecessor_array(predecessor, len(graph))
  if predecessor is None:
    return False

  active_nodes = set(graph.nodes())
  return _valid_forest_simple(graph, predecessor, active_nodes)


def _as_predecessor_array(predecessor, num_nodes):
  """Coerce either a predecessor array or forest adjacency matrix to parents."""
  predecessor = np.asarray(predecessor)
  if predecessor.ndim == 1:
    if len(predecessor) != num_nodes:
      return None
    return predecessor.astype(int)

  if predecessor.ndim != 2 or predecessor.shape != (num_nodes, num_nodes):
    return None

  parents = np.arange(num_nodes)
  for node in range(num_nodes):
    incoming = np.flatnonzero(predecessor[:, node])
    incoming = incoming[incoming != node]
    if len(incoming) > 1:
      return None
    if len(incoming) == 1:
      parents[node] = int(incoming[0])
  return parents


def _valid_forest_simple(graph, predecessor, active_nodes):
  if len(active_nodes) <= 1:
    return True

  subroots = {
      node for node in active_nodes
      if predecessor[node] not in active_nodes or predecessor[node] == node
  }
  if not subroots and active_nodes:
    return False

  subroot_by_node = {}
  if len(subroots) > 1:
    for node in active_nodes:
      subroot = _find_active_subroot(node, predecessor, active_nodes, subroots)
      if subroot is None:
        return False
      subroot_by_node[node] = subroot

    component_graph = nx.DiGraph()
    component_graph.add_nodes_from(subroots)
    for u, v in graph.edges():
      if u not in active_nodes or v not in active_nodes:
        continue
      if subroot_by_node[u] != subroot_by_node[v]:
        component_graph.add_edge(subroot_by_node[u], subroot_by_node[v])

    if not nx.is_directed_acyclic_graph(component_graph):
      return False

  for root in subroots:
    descendants = {
        node for node in active_nodes
        if node != root and _has_active_ancestor(node, root, predecessor, active_nodes)
    }
    if descendants and not _valid_forest_simple(graph, predecessor, descendants):
      return False

  return True


def _find_active_subroot(node, predecessor, active_nodes, subroots):
  seen = set()
  current = node
  while current in active_nodes:
    if current in seen:
      return None
    seen.add(current)
    if current in subroots:
      return current
    current = int(predecessor[current])
  return None


def _has_active_ancestor(node, ancestor, predecessor, active_nodes):
  seen = set()
  current = node
  while current in active_nodes:
    if current in seen:
      return False
    seen.add(current)
    parent = int(predecessor[current])
    if parent == ancestor:
      return True
    if parent == current:
      return False
    current = parent
  return False


def no_self_loops_parent_tree_to_adj_matrix(
    tree):  # FIXME: duplicate code in validate_distributions cuz im lazy
  """now root is just any node without parent"""
  M = parent_tree_to_adj_matrix(tree)
  np.fill_diagonal(a=M, val=0)
  return M


def parent_tree_to_adj_matrix(tree):
  size = len(tree)  # n_vertices
  M = np.zeros((size, size))
  for ix in range(size):
    M[int(tree[ix]), ix] = 1  # edge points tree[ix] to ix, bcuz parent tree
  return M


def preprocess(adj_matrix):
  """ensure it's acyclic, nodes have at most 1 parent"""
  if isinstance(adj_matrix, nx.Graph):
    adj_matrix = nx.to_numpy_array(adj_matrix)
  n = len(adj_matrix)
  # Step 1: Count incoming edges for each node
  in_degrees = np.sum(adj_matrix, axis=0)
  # Step 2: Ensure nodes have no more than 1 parent
  if not np.all((in_degrees == 1) | (in_degrees == 0)):
    return False  # Each node must have either 0 (root) or 1 incoming edge
  # Step 3: Ensure acyclic
  F = nx.DiGraph(adj_matrix)

  return nx.is_forest(F)


def ccv(G, F, node, colors):
  '''check child validity: which kid can go next, like a waterslide'''
  if colors[
      node] == 1:  # you can only be greened by your kids being valid at some point, and once you're green you stay green
    return True
  if down(G, node,
          colors) != treedown(F, node):  #down(F, node, colors):  # base case: this node can go
    return False
  colors[node] = 1  # passed the vibe check, green
  #print(f'greening node {node}')
  kids = F.neighbors(node)
  kids = notgreen(kids, colors)  # green means visited, done
  i = 0
  while i < len(kids):
    kid = kids[i]
    pd = dict()
    ad = dict()
    pd[kid] = down(G, kid, colors)  # possible descendants **through non-green paths**
    ad[kid] = treedown(
        F,
        kid)  #down(F, kid, colors)  # actual descendants **should this be thru non-green? it is**
    #breakpoint()
    if pd[kid] == ad[kid]:
      # this kid can go next
      if not ccv(G, F, kid, colors):  # blow-up if things are bad in the descendants
        #print('fails on descendants')
        return False
      i = 0  # otherwise, proceed to other top-level kids
      kids = notgreen(kids, colors)  # change the loop list
    else:
      i += 1  # this one cant go first. check the next top-level kid

  return notgreen(kids, colors) == [
  ]  # all kids green we gucci, bcuz all subkids green or else return false early


def down(G, source, colors):
  '''return set of all descendents of node in G, cannot pass through green nodes'''
  visited = set()
  descendants = set()

  def notgreen_dfs(node):
    if node in visited or colors[node] == 1:
      return
    visited.add(node)
    descendants.add(node)
    for neighbor in G.neighbors(node):
      notgreen_dfs(neighbor)

  notgreen_dfs(source)
  descendants.discard(
      source)  # Exclude the source itself. If source not present (green at start) does nothing
  return descendants


def treedown(G, source):
  '''return set of all descendents of node in G'''
  visited = set()
  descendants = set()

  def minidfs(node):
    if node in visited:
      return
    visited.add(node)
    descendants.add(node)
    for neighbor in G.neighbors(node):
      minidfs(neighbor)

  minidfs(source)
  descendants.discard(source)  # Exclude the source itself.
  return descendants


def notgreen(kids, color):
  '''kids is a list of node ix'''
  return [kid for kid in kids if color[kid] != 1]


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


def check_valid_dfs_tree(adjacency, parent_tree):
  parent_tree = np.asarray(parent_tree).astype(int)
  return dfsverify(adjacency, parent_tree)


__all__ = (
    "dfsverify_simple",
    "replace_self_loops_with_minus1",
    "are_valid_edges_parents",
    "are_valid_order_parents",
    "is_acyclic",
    "check_valid_dfsTree",
    "check_valid_dfsTree_new",
    "check_valid_dfs_tree",
)
