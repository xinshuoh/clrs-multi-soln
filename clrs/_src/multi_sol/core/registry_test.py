"""Tests for multi-solution extension registry overlays."""

import pathlib
import re

from absl.testing import absltest

from clrs._src import specs
from clrs._src.multi_sol.core import registry


class MultiSolRegistryTest(absltest.TestCase):

  def test_builtin_extensions_registered(self):
    extensions = registry.list_extensions()
    self.assertIn("dfs_multi", extensions)
    self.assertIn("bfs_multi", extensions)
    self.assertIn("bellman_ford_multi", extensions)

  def test_overlay_specs_have_multisol_pi(self):
    resolved_specs = registry.resolve_specs(specs.SPECS)
    self.assertEqual(
        resolved_specs["dfs_multi"]["pi"][2], specs.Type.MULTI_SOLUTION)
    self.assertEqual(
        resolved_specs["bfs_multi"]["pi"][2], specs.Type.MULTI_SOLUTION)
    self.assertEqual(
        resolved_specs["bellman_ford_multi"]["pi"][2], specs.Type.MULTI_SOLUTION)

  def test_multisol_type_alias_is_compatible(self):
    self.assertEqual(specs.Type.MULT_SOL, specs.Type.MULTI_SOLUTION)

  def test_core_modules_do_not_branch_on_multisol_type(self):
    source_root = pathlib.Path(__file__).resolve().parents[2]
    type_branch = re.compile(
        r"(if|elif)\s+.*Type\.(MULT_SOL|MULTI_SOLUTION)")
    for module_name in ("decoders.py", "losses.py", "evaluation.py"):
      with self.subTest(module_name=module_name):
        source = (source_root / module_name).read_text(encoding="utf-8")
        self.assertIsNone(type_branch.search(source))

  def test_extension_algorithms_include_multisol(self):
    extended = registry.get_extension_algorithms(specs.CLRS_30_ALGS)
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      self.assertIn(name, extended)

  def test_builtin_extensions_expose_samplers(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      extension = registry.get_extension(name)
      self.assertIsNotNone(extension)
      self.assertIsNotNone(extension.sampler_factory)

  def test_builtin_extensions_expose_evaluators(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      extension = registry.get_extension(name)
      self.assertIsNotNone(extension)
      self.assertIsNotNone(extension.evaluator)


if __name__ == "__main__":
  absltest.main()
