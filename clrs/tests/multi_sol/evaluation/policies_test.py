"""Tests for pointer-distribution evaluation wiring."""

from absl.testing import absltest

import numpy as np

from clrs._src import evaluation
from clrs._src import probing
from clrs._src import specs


class EvaluationPoliciesTest(absltest.TestCase):

  def test_pointer_distribution_evaluates_in_core(self):
    truth = probing.DataPoint(
        name="pi",
        location=specs.Location.NODE,
        type_=specs.Type.POINTER_DISTRIBUTION,
        data=np.asarray([[[1.0, 0.0], [0.25, 0.75]]]),
    )
    pred = probing.DataPoint(
        name="pi",
        location=specs.Location.NODE,
        type_=specs.Type.POINTER_DISTRIBUTION,
        data=np.asarray([[[1.0, 0.0], [0.75, 0.25]]]),
    )

    scores = evaluation.evaluate((truth,), {"pi": pred})

    self.assertIn("pi", scores)
    self.assertIn("score", scores)
    self.assertAlmostEqual(scores["pi"], 0.75)


if __name__ == "__main__":
  absltest.main()
