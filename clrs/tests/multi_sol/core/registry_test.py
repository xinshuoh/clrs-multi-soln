"""Tests for multi-solution extension registry overlays."""

import pathlib

from absl.testing import absltest

from clrs._src import samplers
from clrs._src import specs
from clrs._src.multi_sol.algorithms.bellman_ford import generator as bf_generator
from clrs._src.multi_sol.algorithms.bfs import generator as bfs_generator
from clrs._src.multi_sol.algorithms.dfs import generator as dfs_generator
from clrs._src.multi_sol.evaluation import pipeline
from clrs._src.multi_sol.core import registry


class MultiSolRegistryTest(absltest.TestCase):

  def test_builtin_extensions_registered(self):
    extensions = registry.list_extensions()
    self.assertIn("dfs_multi", extensions)
    self.assertIn("bfs_multi", extensions)
    self.assertIn("bellman_ford_multi", extensions)

  def test_overlay_specs_have_pointer_distribution_pi(self):
    resolved_specs = registry.resolve_specs(specs.SPECS)
    self.assertEqual(resolved_specs["dfs_multi"]["pi"][2], specs.Type.POINTER_DISTRIBUTION)
    self.assertEqual(resolved_specs["bfs_multi"]["pi"][2], specs.Type.POINTER_DISTRIBUTION)
    self.assertEqual(resolved_specs["bellman_ford_multi"]["pi"][2], specs.Type.POINTER_DISTRIBUTION)

  def test_core_specs_remain_baseline_only(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      self.assertNotIn(name, specs.SPECS)
      self.assertNotIn(name, specs.CLRS_30_ALGS)

  def test_core_type_handling_does_not_import_multisol_policies(self):
    source_root = pathlib.Path(__file__).resolve().parents[3] / "_src"
    for module_name in ("decoders.py", "losses.py", "evaluation.py"):
      with self.subTest(module_name=module_name):
        source = (source_root / module_name).read_text(encoding="utf-8")
        self.assertNotIn("multi_sol.training", source)
        self.assertNotIn("multi_sol.evaluation.output_types", source)

  def test_extension_algorithms_include_multisol(self):
    extended = registry.get_extension_algorithms(specs.CLRS_30_ALGS)
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      self.assertIn(name, extended)

  def test_builtin_extensions_expose_samplers(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      extension = registry.get_extension(name)
      self.assertIsNotNone(extension)
      self.assertIsNotNone(extension.sampler_class)

  def test_builtin_extensions_expose_generators(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      extension = registry.get_extension(name)
      self.assertIsNotNone(extension)
      self.assertIsNotNone(extension.generator)

  def test_builtin_extensions_expose_specs(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      extension = registry.get_extension(name)
      self.assertIsNotNone(extension)
      self.assertIsNotNone(extension.spec)

  def test_multisol_samplers_are_injected_from_multisol_module(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      extension = registry.get_extension(name)
      self.assertIsNotNone(extension)
      sampler_class = extension.sampler_class
      self.assertEqual(sampler_class.__module__, "clrs._src.samplers")
      self.assertIs(samplers.SAMPLERS[name], sampler_class)

  def test_multisol_generators_are_injected_from_multisol_module(self):
    expected_algorithms = {
        "dfs_multi": dfs_generator.dfs_multi,
        "bfs_multi": bfs_generator.bfs_multi,
        "bellman_ford_multi": bf_generator.bellman_ford_multi,
    }
    for name, expected_algorithm in expected_algorithms.items():
      extension = registry.get_extension(name)
      self.assertIsNotNone(extension)
      generator_fn = extension.generator
      self.assertIn(".generator", generator_fn.__module__)
      self.assertIs(generator_fn, expected_algorithm)

  def test_build_sampler_uses_injected_multisol_sampler_classes(self):
    expected_algorithms = {
        "dfs_multi": dfs_generator.dfs_multi,
        "bfs_multi": bfs_generator.bfs_multi,
        "bellman_ford_multi": bf_generator.bellman_ford_multi,
    }
    for name, expected_algorithm in expected_algorithms.items():
      sampler, _ = samplers.build_sampler(name, num_samples=1, length=4, seed=0)
      self.assertEqual(type(sampler).__module__, "clrs._src.samplers")
      self.assertIs(sampler._algorithm, expected_algorithm)  # pylint: disable=protected-access

  def test_builtin_extensions_use_generic_sampling_evaluator(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      extension = registry.get_extension(name)
      self.assertIsNotNone(extension)
      self.assertIsNotNone(pipeline.build_definition_evaluator(extension))

  def test_builtin_extensions_expose_solution_spaces(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      extension = registry.get_extension(name)
      self.assertIsNotNone(extension)
      self.assertIsNotNone(extension.solution_space)
      self.assertIsNotNone(extension.solution_space.batch_extractor)
      self.assertIsNotNone(extension.solution_space.validation_method)
      self.assertNotEmpty(extension.solution_space.extraction_methods)

  def test_builtin_extensions_expose_training_distributions(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      extension = registry.get_extension(name)
      self.assertIsNotNone(extension)
      self.assertEqual(extension.training_distribution.num_solutions, 20)
      self.assertEqual(extension.training_distribution.output_name, "pi")

  def test_builtin_extensions_can_sample_from_algorithm_source(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      extension = registry.get_extension(name)
      self.assertIsNotNone(extension)
      self.assertIsNotNone(extension.solution_space)
      self.assertIsNotNone(extension.solution_space.generator_sampling_source)


if __name__ == "__main__":
  absltest.main()
