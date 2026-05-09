"""Tests for bellman_ford multi-solution generator."""

from absl.testing import absltest

import numpy as np

from clrs._src.multi_sol.algorithms.bellman_ford import generator


X = np.iinfo(np.int32).max  # not connected

# Allow multiple shortest-path solutions:
# edge cost (1 -> 4) is set to 1 instead of 2.
WEIGHTED_DIRECTED_2 = np.array([
    [X, 9, 3, X, X],
    [X, X, 6, X, 1],
    [X, 2, X, 1, X],
    [X, X, 2, X, 2],
    [X, X, X, X, X],
])


class BellmanFordGeneratorTest(absltest.TestCase):

  def test_bellman_ford_multi(self):
    expected = np.array([
        [1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0.],
        [0., 0.6, 0., 0.4, 0.],
    ])
    out, _ = generator.bellman_ford_multi(WEIGHTED_DIRECTED_2, 0, seed=0)
    np.testing.assert_array_equal(expected, out)


if __name__ == "__main__":
  absltest.main()
