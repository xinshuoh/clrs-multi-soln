"""Tests for BFS validators."""

from absl.testing import absltest

import numpy as np

from clrs._src.multi_sol.algorithms.bfs import validator


UNDIRECTED = np.array(
    [
        [0, 1, 0, 0, 1],
        [1, 0, 1, 1, 1],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 1, 0, 1, 0],
    ]
)


class BfsValidationTest(absltest.TestCase):

  def test_accepts_valid_bfs_trees(self):
    self.assertTrue(
        validator.check_valid_bfs_tree(UNDIRECTED, [0, 0, 1, 1, 0], 0))
    self.assertTrue(
        validator.check_valid_bfs_tree(UNDIRECTED, [0, 0, 1, 4, 0], 0))

  def test_accepts_self_parent_for_unreachable_nodes(self):
    graph = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ])
    self.assertTrue(validator.check_valid_bfs_tree(graph, [0, 0, 1, 3], 0))

  def test_rejects_inconsistent_layer_tiebreak(self):
    graph = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 0, 1, 1],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
    ])
    self.assertTrue(validator.check_valid_bfs_tree(graph, [0, 0, 0, 1, 1], 0))
    self.assertTrue(validator.check_valid_bfs_tree(graph, [0, 0, 0, 2, 2], 0))
    self.assertFalse(validator.check_valid_bfs_tree(graph, [0, 0, 0, 1, 2], 0))


if __name__ == "__main__":
  absltest.main()
