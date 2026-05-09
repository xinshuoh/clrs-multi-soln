import types

from absl.testing import absltest

import numpy as np

from clrs._src.multi_sol.algorithms.bfs import extractors


class _DummyDatapoint:
  def __init__(self, data):
    self.data = data


class BfsExtractorsTest(absltest.TestCase):

  def test_extract_prim_simple_path(self):
    prob_matrix = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    batch = types.SimpleNamespace(source_nodes=np.asarray([0]))
    trees = extractors.extract_prim([_DummyDatapoint([prob_matrix])], batch)
    np.testing.assert_array_equal(trees[0], np.array([0, 0, 1]))

  def test_extract_prim_branching(self):
    prob_matrix = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    batch = types.SimpleNamespace(source_nodes=np.asarray([0]))
    trees = extractors.extract_prim([_DummyDatapoint([prob_matrix])], batch)
    np.testing.assert_array_equal(trees[0], np.array([0, 0, 0]))

  def test_extract_prim_disconnected_invariants(self):
    prob_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    batch = types.SimpleNamespace(source_nodes=np.asarray([0]))
    trees = extractors.extract_prim([_DummyDatapoint([prob_matrix])], batch)
    self.assertEqual(trees[0][0], 0)
    self.assertTrue(np.all((trees[0] >= 0) & (trees[0] < 3)))

  def test_extract_categorical_is_deterministic_for_one_hot_rows(self):
    prob_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    trees = extractors.extract_categorical([_DummyDatapoint([prob_matrix])], None)
    np.testing.assert_array_equal(trees[0], np.array([0, 1, 2]))

  def test_extract_beam_prefers_high_mass_parents(self):
    prob_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.95, 0.05, 0.0],
        [0.80, 0.20, 0.0],
    ])
    batch = types.SimpleNamespace(source_nodes=np.asarray([0]))
    trees = extractors.extract_beam(
        [_DummyDatapoint([prob_matrix])], batch, beam_width=2)
    np.testing.assert_array_equal(trees[0], np.array([0, 0, 0]))


if __name__ == "__main__":
  absltest.main()
