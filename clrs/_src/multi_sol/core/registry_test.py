"""Tests for multi-solution extension registry overlays."""

import pathlib
import re

from absl.testing import absltest

from clrs._src import samplers
from clrs._src import specs
from clrs._src.multi_sol import catalog
from clrs._src.multi_sol.algorithms import multi_graphs
from clrs._src.multi_sol.core import registry


class MultiSolRegistryTest(absltest.TestCase):

  def test_builtin_extensions_registered(self):
    extensions = registry.list_extensions()
    self.assertIn("dfs_multi", extensions)
    self.assertIn("bfs_multi", extensions)
    self.assertIn("bellman_ford_multi", extensions)

  def test_overlay_specs_have_pointer_distribution_pi(self):
    resolved_specs = registry.resolve_specs(specs.SPECS)
    self.assertEqual(
        resolved_specs["dfs_multi"]["pi"][2], specs.Type.POINTER_DISTRIBUTION)
    self.assertEqual(
        resolved_specs["bfs_multi"]["pi"][2], specs.Type.POINTER_DISTRIBUTION)
    self.assertEqual(
        resolved_specs["bellman_ford_multi"]["pi"][2],
        specs.Type.POINTER_DISTRIBUTION)

  def test_core_specs_remain_baseline_only(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      self.assertNotIn(name, specs.SPECS)
      self.assertNotIn(name, specs.CLRS_30_ALGS)

  def test_core_modules_do_not_branch_on_multisol_type(self):
    source_root = pathlib.Path(__file__).resolve().parents[2]
    type_branch = re.compile(
        r"(if|elif)\s+.*Type\.POINTER_DISTRIBUTION")
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

  def test_builtin_extensions_expose_algorithm_factories(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      extension = registry.get_extension(name)
      self.assertIsNotNone(extension)
      self.assertIsNotNone(extension.algorithm_factory)

  def test_builtin_extensions_expose_spec_factories(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      extension = registry.get_extension(name)
      self.assertIsNotNone(extension)
      self.assertIsNotNone(extension.spec_factory)
      self.assertIsNone(extension.spec_transformer)

  def test_multisol_samplers_are_injected_from_multisol_module(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      extension = registry.get_extension(name)
      self.assertIsNotNone(extension)
      sampler_class = extension.sampler_factory()
      self.assertEqual(sampler_class.__module__, "clrs._src.multi_sol.samplers")
      self.assertIs(samplers.SAMPLERS[name], sampler_class)

  def test_multisol_algorithms_are_injected_from_multisol_module(self):
    expected_algorithms = {
        "dfs_multi": multi_graphs.dfs_multi,
        "bfs_multi": multi_graphs.bfs_multi,
        "bellman_ford_multi": multi_graphs.bellman_ford_multi,
    }
    for name, expected_algorithm in expected_algorithms.items():
      extension = registry.get_extension(name)
      self.assertIsNotNone(extension)
      algorithm_fn = extension.algorithm_factory()
      self.assertEqual(
          algorithm_fn.__module__, "clrs._src.multi_sol.algorithms.multi_graphs")
      self.assertIs(algorithm_fn, expected_algorithm)

  def test_build_sampler_uses_injected_multisol_sampler_classes(self):
    expected_algorithms = {
        "dfs_multi": multi_graphs.dfs_multi,
        "bfs_multi": multi_graphs.bfs_multi,
        "bellman_ford_multi": multi_graphs.bellman_ford_multi,
    }
    for name, expected_algorithm in expected_algorithms.items():
      sampler, _ = samplers.build_sampler(name, num_samples=1, length=4, seed=0)
      self.assertEqual(type(sampler).__module__, "clrs._src.multi_sol.samplers")
      self.assertIs(sampler._algorithm, expected_algorithm)  # pylint: disable=protected-access

  def test_builtin_extensions_expose_evaluators(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      extension = registry.get_extension(name)
      self.assertIsNotNone(extension)
      self.assertIsNotNone(extension.evaluator)

  def test_registry_builtins_are_catalog_defined(self):
    for extension in catalog.BUILTIN_EXTENSIONS:
      self.assertIs(registry.get_extension(extension.algorithm_name), extension)


if __name__ == "__main__":
  absltest.main()
