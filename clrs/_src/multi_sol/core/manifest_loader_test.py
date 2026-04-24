"""Tests for TOML manifest extension loading."""

from absl.testing import absltest

from clrs._src.multi_sol.core import manifest_loader


class ManifestLoaderTest(absltest.TestCase):

  def test_load_manifest_extensions_contains_builtins(self):
    loaded = manifest_loader.load_manifest_extensions()
    names = {extension.algorithm_name for extension in loaded}
    self.assertIn("dfs_multi", names)
    self.assertIn("bfs_multi", names)
    self.assertIn("bellman_ford_multi", names)

  def test_loaded_extensions_have_direct_symbols(self):
    loaded = manifest_loader.load_manifest_extensions()
    for extension in loaded:
      with self.subTest(name=extension.algorithm_name):
        self.assertIsNotNone(extension.spec)
        self.assertIsNotNone(extension.sampler_class)
        self.assertIsNotNone(extension.algorithm)
        self.assertIsNotNone(extension.evaluator)


if __name__ == "__main__":
  absltest.main()
