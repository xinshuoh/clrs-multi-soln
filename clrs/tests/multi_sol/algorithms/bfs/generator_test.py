"""Tests for BFS multi-solution generator."""

from absl.testing import absltest

import numpy as np

from clrs._src.multi_sol.algorithms.bfs import generator

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


class BfsGeneratorTest(absltest.TestCase):

  def test_bfs_multi_directed(self):
    expected = np.array([
        [1., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1.],
    ])
    out, _ = generator.bfs_multi(DIRECTED, 0, seed=3)
    np.testing.assert_array_equal(expected, out)

  def test_bfs_multi_undirected(self):
    expected = np.array([
        [1., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0.],
        [0., 0.6, 0., 0., 0.4],
        [1., 0., 0., 0., 0.],
    ])
    out, _ = generator.bfs_multi(UNDIRECTED, 0, seed=3)
    np.testing.assert_array_equal(expected, out)


if __name__ == "__main__":
  absltest.main()
