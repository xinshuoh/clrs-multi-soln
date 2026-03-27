"""Tests for extension evaluation policy wiring."""

from absl.testing import absltest

from clrs._src import specs
from clrs._src.multi_sol.evaluation import policies


class EvaluationPoliciesTest(absltest.TestCase):

  def test_multisol_policy_registered(self):
    registered = policies.registered_type_evals()
    self.assertIn(specs.Type.MULTI_SOLUTION, registered)
    self.assertTrue(callable(registered[specs.Type.MULTI_SOLUTION]))
    self.assertIn(specs.Type.MULT_SOL, registered)
    self.assertIs(
        registered[specs.Type.MULT_SOL],
        registered[specs.Type.MULTI_SOLUTION])


if __name__ == "__main__":
  absltest.main()
