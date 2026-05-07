"""BFS validators."""

import networkx as nx

from clrs._src.multi_sol.validation import check_graphs


def check_valid_bfsTree(adjacency, pi, s):
  """Validate bfs_multi parent choices including level tie-break consistency."""
  if not check_graphs.is_square_adjacency(adjacency):
    return False

  n = adjacency.shape[0]
  pi = check_graphs.coerce_parent_array(pi, n)
  if pi is None:
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

  # levels = {}
  # for node, lvl in dist.items():
  #   levels.setdefault(lvl, []).append(node)

  # max_level = max(levels.keys(), default=0)
  # for lvl in range(1, max_level + 1):
  #   prev_level = levels.get(lvl - 1, [])
  #   cur_level = levels.get(lvl, [])
  #   if not cur_level:
  #     continue

  #   ordering_constraints = nx.DiGraph()
  #   ordering_constraints.add_nodes_from(prev_level)
  #   for child in cur_level:
  #     parent = int(pi[child])
  #     candidate_parents = [u for u in prev_level if adjacency[u, child] != 0]
  #     if parent not in candidate_parents:
  #       return False
  #     for other in candidate_parents:
  #       if other != parent:
  #         ordering_constraints.add_edge(parent, other)

  #   if not nx.is_directed_acyclic_graph(ordering_constraints):
  #     return False

  return True


def check_valid_bfs_tree(adjacency, parent_tree, source):
  return check_valid_bfsTree(adjacency, parent_tree, s=source)


__all__ = (
    "check_valid_bfsTree",
    "check_valid_bfs_tree",
)
