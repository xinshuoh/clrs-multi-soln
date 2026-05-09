"""Tests for DFS validators."""

from absl.testing import absltest

import numpy as np
import networkx as nx

from clrs._src.multi_sol.algorithms.dfs.validator import check_valid_dfs_tree


class DfsValidationTest(absltest.TestCase):
  # -------------------------------------------------------------------------
  # Structural Graph Rule Violations
  # -------------------------------------------------------------------------
  def test_invalid_edge_in_forest(self):
    G = nx.DiGraph([(0, 1), (1, 2)])
    # F contains an edge that doesn't exist in G
    F = nx.DiGraph([(0, 1), (0, 2)])
    self.assertFalse(check_valid_dfs_tree(nx.to_numpy_array(G), nx.to_numpy_array(F)))

  def test_cycle_in_forest(self):
    G = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    # F contains a cycle (which violates the preprocess acyclic rule)
    F = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    self.assertFalse(check_valid_dfs_tree(nx.to_numpy_array(G), nx.to_numpy_array(F)))

  def test_multiple_parents_in_forest(self):
    G = nx.DiGraph([(0, 2), (1, 2)])
    # F gives node 2 two parents, which is invalid for a forest
    F = nx.DiGraph([(0, 2), (1, 2)])
    self.assertFalse(check_valid_dfs_tree(nx.to_numpy_array(G), nx.to_numpy_array(F)))

  # -------------------------------------------------------------------------
  # Deep DFS Logic (The "Waterslide" / CCV rules)
  # -------------------------------------------------------------------------
  def test_valid_dfs_chain(self):
    G = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    F = np.array([0, 0, 1, 2])
    self.assertTrue(check_valid_dfs_tree(G, F))

  def test_triangle_valid_path(self):
    # 0 -> 1, 0 -> 2, 1 -> 2
    G = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    # DFS explores 1, then 1 explores 2.
    F_valid = np.array([0, 0, 1])
    self.assertTrue(check_valid_dfs_tree(G, F_valid))

  def test_triangle_invalid_path_skip(self):
    # 0 -> 1, 0 -> 2, 1 -> 2
    G = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    # 0 discovers 1. Then 0 discovers 2.
    # This is INVALID because once 0 discovers 1, 1 has an open path to 2.
    # 1 MUST discover 2 before control returns to 0.
    F_invalid = np.array([0, 1, 0])
    self.assertFalse(check_valid_dfs_tree(G, F_invalid))

  def test_triangle_alternate_valid_path(self):
    # 0 -> 1, 0 -> 2, 1 -> 2
    G = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    # 0 discovers 2 first. 2 has no outgoing edges, so it finishes.
    # Control returns to 0. 0 discovers 1. 1's path to 2 is blocked (2 is green).
    # This is a VALID DFS tree.
    F_valid2 = np.array([0, 0, 0])
    self.assertTrue(check_valid_dfs_tree(G, F_valid2))

  def test_disconnected_components(self):
    # G has two separate components
    G = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    F = np.array([0, 0, 2, 2])
    self.assertTrue(check_valid_dfs_tree(G, F))

  # -------------------------------------------------------------------------
  # Sanity Check Matrices from the Snippet
  # -------------------------------------------------------------------------
  def test_author_tricky_example_A_T(self):
    # These are the matrices the author explicitly flagged in Snippet 1
    A = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0], [1, 1, 0, 0, 0, 1],
                  [0, 1, 1, 0, 0, 1], [0, 1, 0, 0, 0, 0]])

    T = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1],
                  [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])

    self.assertFalse(check_valid_dfs_tree(A, T))

  def test_author_tricky_example_A3_T3(self):
    A3 = np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]])
    T3 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])

    self.assertTrue(check_valid_dfs_tree(A3, T3))


if __name__ == '__main__':
  absltest.main()
