"""BFS validators."""

from clrs._src.algorithms import check_graphs


def check_valid_bfs_tree(adjacency, parent_tree, source):
  return check_graphs.check_valid_bfsTree(adjacency, parent_tree, s=source)

