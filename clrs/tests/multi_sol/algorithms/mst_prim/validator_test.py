"""Tests for MST-Prim validators."""

from absl.testing import absltest

import numpy as np

from clrs._src.multi_sol.algorithms.mst_prim import validator


class MstPrimValidationTest(absltest.TestCase):

  def test_accepts_minimum_spanning_tree(self):
    adjacency = np.array([
        [0, 1, 2],
        [1, 0, 1],
        [2, 1, 0],
    ])
    self.assertTrue(
        validator.check_valid_mst_prim_tree(adjacency, [0, 0, 1], 0))

  def test_accepts_alternative_minimum_spanning_tree(self):
    adjacency = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ])
    self.assertTrue(
        validator.check_valid_mst_prim_tree(adjacency, [0, 2, 0], 0))

  def test_rejects_non_minimum_spanning_tree(self):
    adjacency = np.array([
        [0, 1, 2],
        [1, 0, 1],
        [2, 1, 0],
    ])
    self.assertFalse(
        validator.check_valid_mst_prim_tree(adjacency, [0, 0, 0], 0))

  def test_rejects_parent_cycle(self):
    adjacency = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ])
    self.assertFalse(
        validator.check_valid_mst_prim_tree(adjacency, [0, 2, 1], 0))

  def test_rejects_missing_parent_edge(self):
    adjacency = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ])
    self.assertFalse(
        validator.check_valid_mst_prim_tree(adjacency, [0, 0, 0], 0))

  def test_accepts_disconnected_graph_with_self_parent_unreachable(self):
    adjacency = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    self.assertTrue(
        validator.check_valid_mst_prim_tree(adjacency, [0, 0, 2], 0))

  def test_rejects_non_self_parent_unreachable_node(self):
    adjacency = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    self.assertFalse(
        validator.check_valid_mst_prim_tree(adjacency, [0, 0, 1], 0))


if __name__ == "__main__":
  absltest.main()
