"""Tests for dfs multi-solution generator."""

from absl.testing import absltest

import numpy as np

from clrs._src.multi_sol.algorithms.dfs import generator

# Directed graph
# Looks like:
#     0
#   /  \
#  v    v
#  1 <- 3
#   \  ^
#   v /
#    4 <- 2 -> 5 (self loop)
DIRECTED = np.array([
    [0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1],
])

# Undirected graph
# Looks like:
# 0 - 1 - 2
#  \ / \ /
#   4 - 3
UNDIRECTED = np.array([
    [0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 1, 0, 1, 0],
])


class DfsGeneratorTest(absltest.TestCase):

  def test_dfs_multi_directed(self):
    expected = np.array([
        [1., 0., 0., 0., 0., 0.],
        [0.5, 0., 0., 0.5, 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0.5, 0., 0., 0., 0.5, 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
    ])
    out, _ = generator.dfs_multi(DIRECTED, seed=3)
    np.testing.assert_array_equal(expected, out)

  def test_dfs_multi_undirected(self):
    expected = np.array([
        [1., 0., 0., 0., 0.],
        [0.45, 0., 0.2, 0.15, 0.2],
        [0., 0.45, 0., 0.55, 0.],
        [0., 0.3, 0.3, 0., 0.4],
        [0.55, 0.05, 0., 0.4, 0.],
    ])
    out, _ = generator.dfs_multi(UNDIRECTED, seed=3)
    np.testing.assert_array_equal(expected, out)


if __name__ == "__main__":
  absltest.main()
