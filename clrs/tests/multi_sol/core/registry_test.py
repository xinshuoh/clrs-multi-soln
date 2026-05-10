"""Tests for multi-solution metadata registry."""

import pathlib

from absl.testing import absltest

from clrs._src import algorithms
from clrs._src import samplers
from clrs._src import specs
from clrs._src.multi_sol.evaluation import pipeline
from clrs._src.multi_sol.core import registry


class MultiSolRegistryTest(absltest.TestCase):

  def test_builtin_extensions_registered(self):
    extensions = registry.names()
    self.assertIn("dfs_multi", extensions)
    self.assertIn("bfs_multi", extensions)
    self.assertIn("bellman_ford_multi", extensions)

  def test_core_specs_include_multisol_entries(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      self.assertIn(name, specs.SPECS)
      self.assertEqual(
          specs.SPECS[name]["pi"][2], specs.Type.POINTER_DISTRIBUTION)

  def test_core_type_handling_does_not_import_multisol_policies(self):
    source_root = pathlib.Path(__file__).resolve().parents[3] / "_src"
    for module_name in ("decoders.py", "losses.py", "evaluation.py"):
      with self.subTest(module_name=module_name):
        source = (source_root / module_name).read_text(encoding="utf-8")
        self.assertNotIn("multi_sol.training", source)
        self.assertNotIn("multi_sol.evaluation.output_types", source)

  def test_multisol_samplers_are_registered_directly(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      self.assertIn(name, samplers.SAMPLERS)
      self.assertEqual(samplers.SAMPLERS[name].__module__, "clrs._src.samplers")

  def test_multisol_generators_are_exposed_in_algorithms_namespace(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      generator_fn = getattr(algorithms, name)
      self.assertIn(".generator", generator_fn.__module__)

  def test_build_sampler_uses_algorithms_namespace(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      sampler, _ = samplers.build_sampler(name, num_samples=1, length=4, seed=0)
      self.assertIs(sampler._algorithm, getattr(algorithms, name))  # pylint: disable=protected-access

  def test_builtin_extensions_expose_solution_spaces(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      extension = registry.get(name)
      self.assertIsNotNone(extension)
      self.assertEqual(extension.name, name)
      self.assertIsNotNone(extension.solution_space.batch_extractor)
      self.assertIsNotNone(extension.solution_space.validator)
      self.assertNotEmpty(extension.solution_space.extraction_methods)

  def test_builtin_extensions_expose_training_config(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      extension = registry.get(name)
      self.assertIsNotNone(extension)
      self.assertEqual(extension.training.num_solutions, 20)
      self.assertEqual(extension.training.output_name, "pi")
      self.assertIsNotNone(extension.training.symbolic_sampler)

  def test_builtin_extensions_expose_algorithm_baselines(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      extension = registry.get(name)
      self.assertIsNotNone(extension)
      self.assertIsNotNone(extension.solution_space.algorithm_baseline)

  def test_builtin_extensions_use_generic_sampling_evaluator(self):
    for name in ("dfs_multi", "bfs_multi", "bellman_ford_multi"):
      extension = registry.get(name)
      self.assertIsNotNone(extension)
      self.assertIsNotNone(pipeline.build_definition_evaluator(extension))


if __name__ == "__main__":
  absltest.main()
