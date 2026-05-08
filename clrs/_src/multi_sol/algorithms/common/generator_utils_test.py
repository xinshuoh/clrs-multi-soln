"""Tests for shared multi-solution target-generation helpers."""

from absl.testing import absltest
import numpy as np

from clrs._src.multi_sol.algorithms.common import generator_utils


class CommonTargetGenerationTest(absltest.TestCase):

  def test_generate_parent_distribution_target_repeats_and_patches_probe(self):
    probes_seen = []

    def run_single(_rng, algorithm_spec, deterministic):
      self.assertIsNotNone(algorithm_spec)
      self.assertFalse(deterministic)
      index = len(probes_seen)
      probes = {"output": {"node": {"pi": {"data": None}}}}
      probes_seen.append(probes)
      return np.asarray([0, index % 2]), probes

    parent_dist, probes = generator_utils.generate_parent_distribution_target(
        algorithm_name="bfs_multi",
        num_nodes=2,
        seed=7,
        deterministic=False,
        run_single=run_single,
        num_solutions=4,
    )

    expected = np.asarray([
        [1.0, 0.0],
        [0.5, 0.5],
    ])
    np.testing.assert_array_equal(parent_dist, expected)
    self.assertIs(probes, probes_seen[0])
    np.testing.assert_array_equal(
        probes["output"]["node"]["pi"]["data"], expected)
    self.assertLen(probes_seen, 4)

  def test_generate_parent_distribution_target_deterministic_runs_once(self):
    calls = []

    def run_single(_rng, _algorithm_spec, deterministic):
      calls.append(deterministic)
      probes = {"output": {"node": {"pi": {"data": None}}}}
      return np.asarray([0, 0]), probes

    parent_dist, _ = generator_utils.generate_parent_distribution_target(
        algorithm_name="bfs_multi",
        num_nodes=2,
        seed=7,
        deterministic=True,
        run_single=run_single,
        num_solutions=4,
    )

    np.testing.assert_array_equal(parent_dist, np.asarray([[1.0, 0.0],
                                                          [1.0, 0.0]]))
    self.assertEqual(calls, [True])


if __name__ == "__main__":
  absltest.main()
