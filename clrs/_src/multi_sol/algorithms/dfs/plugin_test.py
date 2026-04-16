import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np


_MISSING = object()


def _find_repo_root() -> pathlib.Path:
  cur = pathlib.Path(__file__).resolve()
  for parent in [cur] + list(cur.parents):
    if (parent / "setup.py").exists():
      return parent
  raise RuntimeError("Could not locate repository root from test path.")


class DummyDataPoint:

  def __init__(self, data):
    self.data = np.asarray(data)


class DummyFeedback:

  def __init__(self, outputs, adjacency):
    self.outputs = outputs
    self.features = object()
    self._adjacency = adjacency

  def __getitem__(self, idx):
    if idx != 0:
      raise IndexError(idx)
    return [[None, DummyDataPoint(self._adjacency)]]


class DfsPluginTest(unittest.TestCase):

  def setUp(self):
    self._saved_modules = {}
    self._repo_root = _find_repo_root()
    self.addCleanup(self._restore_modules)

  def _restore_modules(self):
    for name, old_value in self._saved_modules.items():
      if old_value is _MISSING:
        sys.modules.pop(name, None)
      else:
        sys.modules[name] = old_value

  def _install_module(self, name, module):
    if name not in self._saved_modules:
      self._saved_modules[name] = sys.modules.get(name, _MISSING)
    sys.modules[name] = module
    if "." in name:
      parent_name, child_name = name.rsplit(".", 1)
      parent = sys.modules.get(parent_name)
      if parent is not None:
        setattr(parent, child_name, module)

  def _install_package(self, name):
    module = types.ModuleType(name)
    module.__path__ = []
    self._install_module(name, module)
    return module

  def _load_plugin_with_stubs(self):
    clrs_pkg = self._install_package("clrs")
    self._install_package("clrs._src")
    self._install_package("clrs._src.multi_sol")
    self._install_package("clrs._src.multi_sol.algorithms")
    self._install_package("clrs._src.multi_sol.algorithms.dfs")
    self._install_package("clrs._src.multi_sol.data")
    self._install_package("clrs._src.multi_sol.evaluation")
    self._install_package("clrs._src.multi_sol.sampling")
    self._install_package("clrs._src.multi_sol.validation")

    clrs_pkg.evaluate = lambda outputs, preds: {"score": np.asarray(0.66)}

    jax_module = types.ModuleType("jax")
    jax_module.random = types.SimpleNamespace(split=lambda key: (key + 1, key + 2))
    self._install_module("jax", jax_module)

    adapters_module = types.ModuleType("clrs._src.multi_sol.data.adapters")
    adapters_module.concat_tree = lambda items, axis: items[0]
    self._install_module("clrs._src.multi_sol.data.adapters", adapters_module)

    dist_validation_module = types.ModuleType(
        "clrs._src.multi_sol.evaluation.distribution_validation")
    dist_validation_module.run_dfs_distribution_validation = (
        lambda **kwargs: None)
    self._install_module(
        "clrs._src.multi_sol.evaluation.distribution_validation",
        dist_validation_module)

    reports_module = types.ModuleType("clrs._src.multi_sol.evaluation.reports")
    reports_module.build_dfs_result_dict = lambda **kwargs: {"base": True}
    self._install_module("clrs._src.multi_sol.evaluation.reports", reports_module)

    runners_module = types.ModuleType("clrs._src.multi_sol.evaluation.runners")
    runners_module.evaluate_sampling_pair = lambda **kwargs: {
        "model_trees": [],
        "true_trees": [],
        "model_mask": [],
        "true_mask": [],
        "model_accuracy": 0.0,
        "true_accuracy": 0.0,
    }
    self._install_module("clrs._src.multi_sol.evaluation.runners", runners_module)

    dfs_sampling_module = types.ModuleType("clrs._src.multi_sol.sampling.dfs")
    dfs_sampling_module.sample_random_list = lambda data: []
    dfs_sampling_module.sample_argmax_listofdict = lambda data: []
    dfs_sampling_module.sample_argmax_listofdatapoint = lambda data: []
    dfs_sampling_module.sample_upwards = lambda data: []
    dfs_sampling_module.sample_altUpwards = lambda data: []
    self._install_module("clrs._src.multi_sol.sampling.dfs", dfs_sampling_module)

    dfs_validation_module = types.ModuleType("clrs._src.multi_sol.validation.dfs")
    dfs_validation_module.check_valid_dfs_tree = (
        lambda adjacency, parent_tree, source: True)
    self._install_module("clrs._src.multi_sol.validation.dfs",
                         dfs_validation_module)

    uniqueness_module = types.ModuleType("clrs._src.dfs_uniqueness_check")
    uniqueness_module.check_uniqueness_dfs = lambda *args, **kwargs: (
        0.1, 0.2, 0.3)
    self._install_module("clrs._src.dfs_uniqueness_check", uniqueness_module)

    plugin_path = (
        self._repo_root / "clrs" / "_src" / "multi_sol" / "algorithms" /
        "dfs" / "plugin.py")
    spec = importlib.util.spec_from_file_location(
        "clrs._src.multi_sol.algorithms.dfs.plugin", str(plugin_path))
    module = importlib.util.module_from_spec(spec)
    self._install_module("clrs._src.multi_sol.algorithms.dfs.plugin", module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

  def test_distribution_validation_runs_when_enabled(self):
    plugin = self._load_plugin_with_stubs()

    captured_validation = {}
    plugin.distribution_validation.run_dfs_distribution_validation = (
        lambda **kwargs: captured_validation.update(kwargs))

    saved = {}

    def save_results_fn(result_dict, filename):
      saved["result_dict"] = result_dict
      saved["filename"] = filename

    adjacency = np.asarray([[[0, 1], [1, 0]]])
    feedback = DummyFeedback(
        outputs=[DummyDataPoint(np.asarray([[[1.0, 0.0], [1.0, 0.0]]]))],
        adjacency=adjacency,
    )

    class SingleSampler:
      def __iter__(self):
        return self

      def __next__(self):
        return feedback

    def predict_fn(rng_key, features):
      del rng_key, features
      return {"pi": DummyDataPoint(np.asarray([[[1.0, 0.0], [1.0, 0.0]]]))}, None

    out = plugin.evaluate_dfs_multisol_batch(
        sampler=iter(SingleSampler()),
        predict_fn=predict_fn,
        sample_count=1,
        rng_key=0,
        extras={"phase": "ok"},
        save_results_fn=save_results_fn,
        filename="dfs_case",
        vd_flag=True,
        NSE=42,
        output_dir="results/run-7",
    )

    self.assertEqual(saved["filename"], "dfs_case_DFS")
    self.assertEqual(out["score"], 0.66)
    self.assertEqual(out["phase"], "ok")
    self.assertEqual(captured_validation["nse"], 42)
    self.assertEqual(captured_validation["output_dir"], "results/run-7")
    np.testing.assert_array_equal(captured_validation["adjacency"], adjacency)


if __name__ == "__main__":
  unittest.main()
