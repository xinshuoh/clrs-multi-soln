"""Bellman-Ford validators."""

from clrs._src.algorithms import check_graphs


def check_valid_bf_paths(adjacency, parent_tree, source):
  return check_graphs.check_valid_BFpaths(adjacency, source, parent_tree)

