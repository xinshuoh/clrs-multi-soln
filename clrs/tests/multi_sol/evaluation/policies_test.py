"""Tests for extension evaluation policy wiring."""

from absl.testing import absltest

from clrs._src import specs
from clrs._src.multi_sol.evaluation import policies


class EvaluationPoliciesTest(absltest.TestCase):

  def test_pointer_distribution_policy_registered(self):
    registered = policies.registered_type_evals()
    self.assertIn(specs.Type.POINTER_DISTRIBUTION, registered)
    self.assertTrue(callable(registered[specs.Type.POINTER_DISTRIBUTION]))


if __name__ == "__main__":
  absltest.main()
