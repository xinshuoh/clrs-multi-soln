"""DFS validators."""

import numpy as np

from clrs._src.algorithms import dfs_verification_tester


def check_valid_dfs_tree(adjacency, parent_tree, _source_unused):
  parent_tree = np.asarray(parent_tree).astype(int)
  return dfs_verification_tester.dfsverify(adjacency, parent_tree)
