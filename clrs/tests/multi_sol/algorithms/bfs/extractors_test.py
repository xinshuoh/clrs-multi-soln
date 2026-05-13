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

  def test_extract_prim_self_parents_when_no_processed_mass(self):
    prob_matrix = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    batch = types.SimpleNamespace(source_nodes=np.asarray([0]))
    trees = extractors.extract_prim([_DummyDatapoint([prob_matrix])], batch)
    np.testing.assert_array_equal(trees[0], np.array([0, 0, 2]))

  def test_extract_wave_expands_predicted_waves(self):
    prob_matrix = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    batch = types.SimpleNamespace(source_nodes=np.asarray([0]))
    trees = extractors.extract_wave([_DummyDatapoint([prob_matrix])], batch)
    np.testing.assert_array_equal(trees[0], np.array([0, 0, 1]))

  def test_extract_wave_self_parents_unreached_nodes(self):
    prob_matrix = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    batch = types.SimpleNamespace(source_nodes=np.asarray([0]))
    trees = extractors.extract_wave([_DummyDatapoint([prob_matrix])], batch)
    np.testing.assert_array_equal(trees[0], np.array([0, 0, 2]))

  def test_extract_wave_self_parents_when_self_mass_dominates_reached_mass(self):
    prob_matrix = np.array([
        [0.8, 0.05, 0.05, 0.05, 0.05],
        [0.1, 0.6, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.6, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.6, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.6],
    ])
    batch = types.SimpleNamespace(source_nodes=np.asarray([0]))
    trees = extractors.extract_wave([_DummyDatapoint([prob_matrix])], batch)
    np.testing.assert_array_equal(trees[0], np.array([0, 1, 2, 3, 4]))

  def test_extract_wave_discovers_when_reached_mass_exceeds_self_mass(self):
    prob_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.6, 0.4, 0.0],
        [0.0, 0.6, 0.4],
    ])
    batch = types.SimpleNamespace(source_nodes=np.asarray([0]))
    trees = extractors.extract_wave([_DummyDatapoint([prob_matrix])], batch)
    np.testing.assert_array_equal(trees[0], np.array([0, 0, 1]))

  def test_extract_wave_samples_parent_from_reached_set(self):
    prob_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5, 0.0],
    ])
    sampled_parents = set()
    for seed in range(10):
      np.random.seed(seed)
      sampled_parents.add(extractors._wave_sampler(prob_matrix, 0)[3])

    self.assertIn(1, sampled_parents)
    self.assertIn(2, sampled_parents)

  def test_extract_categorical_is_deterministic_for_one_hot_rows(self):
    prob_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    batch = types.SimpleNamespace(source_nodes=np.asarray([0]))
    trees = extractors.extract_categorical([_DummyDatapoint([prob_matrix])], batch)
    np.testing.assert_array_equal(trees[0], np.array([0, 1, 2]))

  def test_extract_categorical_keeps_source_self_parent(self):
    prob_matrix = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    batch = types.SimpleNamespace(source_nodes=np.asarray([0]))
    trees = extractors.extract_categorical([_DummyDatapoint([prob_matrix])], batch)
    np.testing.assert_array_equal(trees[0], np.array([0, 0, 1]))

  def test_extract_categorical_self_parents_zero_mass_rows(self):
    prob_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    batch = types.SimpleNamespace(source_nodes=np.asarray([0]))
    trees = extractors.extract_categorical([_DummyDatapoint([prob_matrix])], batch)
    np.testing.assert_array_equal(trees[0], np.array([0, 1, 1]))

  def test_extract_beam_prefers_high_mass_parents(self):
    prob_matrix = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    batch = types.SimpleNamespace(source_nodes=np.asarray([0]))
    trees = extractors.extract_beam(
        [_DummyDatapoint([prob_matrix])], batch, beam_width=2)
    np.testing.assert_array_equal(trees[0], np.array([0, 0, 0]))

  def test_extract_beam_keeps_source_self_parent(self):
    prob_matrix = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
    ])
    batch = types.SimpleNamespace(source_nodes=np.asarray([1]))
    trees = extractors.extract_beam(
        [_DummyDatapoint([prob_matrix])], batch, beam_width=2)
    self.assertEqual(trees[0][1], 1)

  def test_extract_beam_self_parents_when_no_processed_mass(self):
    prob_matrix = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    batch = types.SimpleNamespace(source_nodes=np.asarray([0]))
    trees = extractors.extract_beam(
        [_DummyDatapoint([prob_matrix])], batch, beam_width=2)
    np.testing.assert_array_equal(trees[0], np.array([0, 0, 2]))

  def test_extract_beam_samples_parent_candidates(self):
    prob_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    sampled_trees = set()
    for seed in range(10):
      np.random.seed(seed)
      sampled_trees.add(
          tuple(extractors._bfs_beam_sampler(prob_matrix, 0, beam_width=1)))

    self.assertIn((0, 0, 0, 2), sampled_trees)
    self.assertIn((0, 0, 1, 2), sampled_trees)

  def test_active_bfs_extractors_include_wave_not_beam(self):
    self.assertEqual(
        [extractor.name for extractor in extractors.EXTRACTORS],
        ["Categorical", "Random", "Prim", "Wave"],
    )


if __name__ == "__main__":
  absltest.main()
