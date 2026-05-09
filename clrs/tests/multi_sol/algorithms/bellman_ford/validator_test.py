"""Tests for Bellman-Ford validators."""

from absl.testing import absltest

import numpy as np

from clrs._src.multi_sol.algorithms.bellman_ford import validator


WEIGHTED_DIRECTED = np.array(
    [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 2, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0],
    ]
)

DISCONNECTED = np.array([
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 0],
])


class BellmanFordValidationTest(absltest.TestCase):

  def test_accepts_shortest_path_tree(self):
    self.assertTrue(
        validator.check_valid_bf_paths(
            WEIGHTED_DIRECTED, [0, 0, 1, 1, 2], 0))

  def test_rejects_non_shortest_path_tree(self):
    self.assertFalse(
        validator.check_valid_bf_paths(
            WEIGHTED_DIRECTED, [0, 0, 1, 1, 3], 0))

  def test_rejects_nonexistent_parent_edge(self):
    self.assertFalse(
        validator.check_valid_bf_paths(
            WEIGHTED_DIRECTED, [0, 2, 0, 1, 2], 0))

  def test_accepts_unreachable_node_self_parent(self):
    self.assertTrue(validator.check_valid_bf_paths(DISCONNECTED, [0, 0, 2], 0))

  def test_bellman_ford_cost(self):
    expected = np.array([0, 1, 2, 3, 3])
    actual = validator.bellman_ford_cost(WEIGHTED_DIRECTED, 0)
    np.testing.assert_array_equal(expected, actual)


if __name__ == "__main__":
  absltest.main()
