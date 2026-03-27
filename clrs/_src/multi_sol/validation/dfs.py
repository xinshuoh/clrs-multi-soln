"""DFS validators."""

from clrs._src.algorithms import dfs_verification_tester


def check_valid_dfs_tree(adjacency, parent_tree, _source_unused):
  return dfs_verification_tester.dfsverify(adjacency, parent_tree)

