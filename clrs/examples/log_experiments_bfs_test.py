import importlib.util
import pathlib
import sys
import types
import unittest
import warnings


_MISSING = object()


def _find_repo_root() -> pathlib.Path:
  cur = pathlib.Path(__file__).resolve()
  for parent in [cur] + list(cur.parents):
    if (parent / "setup.py").exists():
      return parent
  raise RuntimeError("Could not locate repository root from test path.")


class LogExperimentsWrapperTest(unittest.TestCase):

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

  def _load_log_experiments(self, bfs_plugin, dfs_plugin, bf_plugin):
    self._install_package("clrs")
    self._install_package("clrs._src")
    self._install_package("clrs._src.multi_sol")
    self._install_package("clrs._src.multi_sol.algorithms")
    self._install_package("clrs._src.multi_sol.algorithms.bfs")
    self._install_package("clrs._src.multi_sol.algorithms.dfs")
    self._install_package("clrs._src.multi_sol.algorithms.bellman_ford")
    self._install_package("clrs._src.multi_sol.evaluation")

    class _DataFrameStub:

      @staticmethod
      def from_dict(_data):
        return types.SimpleNamespace(to_csv=lambda *args, **kwargs: None)

    pandas_module = types.ModuleType("pandas")
    pandas_module.DataFrame = _DataFrameStub
    self._install_module("pandas", pandas_module)

    reporting_module = types.ModuleType("clrs._src.multi_sol.evaluation.reporting")
    reporting_module.save_csv_report = lambda *args, **kwargs: None
    self._install_module("clrs._src.multi_sol.evaluation.reporting",
                         reporting_module)

    bfs_plugin_module = types.ModuleType("clrs._src.multi_sol.algorithms.bfs.plugin")
    bfs_plugin_module.evaluate_bfs_multisol_batch = bfs_plugin
    self._install_module("clrs._src.multi_sol.algorithms.bfs.plugin",
                         bfs_plugin_module)
    dfs_plugin_module = types.ModuleType("clrs._src.multi_sol.algorithms.dfs.plugin")
    dfs_plugin_module.evaluate_dfs_multisol_batch = dfs_plugin
    self._install_module("clrs._src.multi_sol.algorithms.dfs.plugin",
                         dfs_plugin_module)
    bf_plugin_module = types.ModuleType(
        "clrs._src.multi_sol.algorithms.bellman_ford.plugin")
    bf_plugin_module.evaluate_bf_multisol_batch = bf_plugin
    self._install_module("clrs._src.multi_sol.algorithms.bellman_ford.plugin",
                         bf_plugin_module)

    module_path = self._repo_root / "clrs" / "examples" / "log_experiments.py"
    spec = importlib.util.spec_from_file_location("clrs.examples.log_experiments",
                                                  str(module_path))
    module = importlib.util.module_from_spec(spec)
    self._install_module("clrs.examples", self._install_package("clrs.examples"))
    self._install_module("clrs.examples.log_experiments", module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

  def _assert_deprecation_warning(self, caught, symbol):
    self.assertTrue(
        any(symbol in str(warning.message) for warning in caught),
        msg=f"Expected deprecation warning mentioning `{symbol}`.",
    )

  def test_bfs_wrapper_delegates_to_plugin(self):
    captured = {}

    def bfs_plugin(**kwargs):
      captured.update(kwargs)
      return {"delegated_bfs": 1.0}

    module = self._load_log_experiments(
        bfs_plugin=bfs_plugin,
        dfs_plugin=lambda **kwargs: {"delegated_dfs": 1.0},
        bf_plugin=lambda **kwargs: {"delegated_bf": 1.0},
    )
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      result = module.BFS_multi_collect_and_eval(
          sampler="sampler",
          predict_fn="predict",
          sample_count=7,
          rng_key=123,
          extras={"extra": True},
          filename="regression",
          vd_flag=True,
          NSE=11,
      )

    self.assertEqual(result, {"delegated_bfs": 1.0})
    self.assertEqual(captured["filename"], "regression")
    self.assertIs(captured["save_results_fn"], module.save_results)
    self.assertNotIn("vd_flag", captured)
    self.assertNotIn("NSE", captured)
    self._assert_deprecation_warning(caught, "BFS_multi_collect_and_eval")

  def test_dfs_wrapper_forwards_validation_args(self):
    captured = {}

    def dfs_plugin(**kwargs):
      captured.update(kwargs)
      return {"delegated_dfs": 1.0}

    module = self._load_log_experiments(
        bfs_plugin=lambda **kwargs: {"delegated_bfs": 1.0},
        dfs_plugin=dfs_plugin,
        bf_plugin=lambda **kwargs: {"delegated_bf": 1.0},
    )
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      result = module.DFS_collect_and_eval(
          sampler="sampler",
          predict_fn="predict",
          sample_count=9,
          rng_key=456,
          extras={"flag": "x"},
          filename="dfs_run",
          vd_flag=True,
          NSE=33,
      )

    self.assertEqual(result, {"delegated_dfs": 1.0})
    self.assertEqual(captured["filename"], "dfs_run")
    self.assertTrue(captured["vd_flag"])
    self.assertEqual(captured["NSE"], 33)
    self.assertIs(captured["save_results_fn"], module.save_results)
    self._assert_deprecation_warning(caught, "DFS_collect_and_eval")

  def test_bf_wrapper_forwards_validation_args(self):
    captured = {}

    def bf_plugin(**kwargs):
      captured.update(kwargs)
      return {"delegated_bf": 1.0}

    module = self._load_log_experiments(
        bfs_plugin=lambda **kwargs: {"delegated_bfs": 1.0},
        dfs_plugin=lambda **kwargs: {"delegated_dfs": 1.0},
        bf_plugin=bf_plugin,
    )
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      result = module.BF_collect_and_eval(
          sampler="sampler",
          predict_fn="predict",
          sample_count=5,
          rng_key=789,
          extras={"flag": "y"},
          filename="bf_run",
          vd_flag=False,
          NSE=22,
      )

    self.assertEqual(result, {"delegated_bf": 1.0})
    self.assertEqual(captured["filename"], "bf_run")
    self.assertFalse(captured["vd_flag"])
    self.assertEqual(captured["NSE"], 22)
    self.assertIs(captured["save_results_fn"], module.save_results)
    self._assert_deprecation_warning(caught, "BF_collect_and_eval")

  def test_bfs_wrapper_allows_explicit_sink_override(self):
    captured = {}

    def bfs_plugin(**kwargs):
      captured.update(kwargs)
      return {"delegated_bfs": 1.0}

    module = self._load_log_experiments(
        bfs_plugin=bfs_plugin,
        dfs_plugin=lambda **kwargs: {"delegated_dfs": 1.0},
        bf_plugin=lambda **kwargs: {"delegated_bf": 1.0},
    )
    custom_sink = lambda *_args, **_kwargs: None
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      module.BFS_multi_collect_and_eval(
          sampler="sampler",
          predict_fn="predict",
          sample_count=3,
          rng_key=99,
          extras={},
          filename="regression",
          save_results_fn=custom_sink,
      )
    self.assertIs(captured["save_results_fn"], custom_sink)
    self._assert_deprecation_warning(caught, "BFS_multi_collect_and_eval")

  def test_save_results_delegates_to_reporting_sink(self):
    writes = {}

    self._install_package("clrs")
    self._install_package("clrs._src")
    self._install_package("clrs._src.multi_sol")
    self._install_package("clrs._src.multi_sol.algorithms")
    self._install_package("clrs._src.multi_sol.algorithms.bfs")
    self._install_package("clrs._src.multi_sol.algorithms.dfs")
    self._install_package("clrs._src.multi_sol.algorithms.bellman_ford")
    self._install_package("clrs._src.multi_sol.evaluation")

    reporting_module = types.ModuleType("clrs._src.multi_sol.evaluation.reporting")
    reporting_module.save_csv_report = (
        lambda result_dict, filename: writes.update({
            "result_dict": result_dict,
            "filename": filename,
        }))
    self._install_module("clrs._src.multi_sol.evaluation.reporting",
                         reporting_module)

    bfs_plugin_module = types.ModuleType("clrs._src.multi_sol.algorithms.bfs.plugin")
    bfs_plugin_module.evaluate_bfs_multisol_batch = lambda **kwargs: {}
    self._install_module("clrs._src.multi_sol.algorithms.bfs.plugin",
                         bfs_plugin_module)
    dfs_plugin_module = types.ModuleType("clrs._src.multi_sol.algorithms.dfs.plugin")
    dfs_plugin_module.evaluate_dfs_multisol_batch = lambda **kwargs: {}
    self._install_module("clrs._src.multi_sol.algorithms.dfs.plugin",
                         dfs_plugin_module)
    bf_plugin_module = types.ModuleType(
        "clrs._src.multi_sol.algorithms.bellman_ford.plugin")
    bf_plugin_module.evaluate_bf_multisol_batch = lambda **kwargs: {}
    self._install_module("clrs._src.multi_sol.algorithms.bellman_ford.plugin",
                         bf_plugin_module)

    module_path = self._repo_root / "clrs" / "examples" / "log_experiments.py"
    spec = importlib.util.spec_from_file_location("clrs.examples.log_experiments",
                                                  str(module_path))
    module = importlib.util.module_from_spec(spec)
    self._install_module("clrs.examples", self._install_package("clrs.examples"))
    self._install_module("clrs.examples.log_experiments", module)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      module.save_results({"a": [1]}, "sample")

    self.assertEqual(writes["result_dict"], {"a": [1]})
    self.assertEqual(writes["filename"], "sample")
    self._assert_deprecation_warning(caught, "save_results")

  def test_main_exits_with_deprecation_guidance(self):
    module = self._load_log_experiments(
        bfs_plugin=lambda **kwargs: {"delegated_bfs": 1.0},
        dfs_plugin=lambda **kwargs: {"delegated_dfs": 1.0},
        bf_plugin=lambda **kwargs: {"delegated_bf": 1.0},
    )

    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      with self.assertRaises(SystemExit) as ctx:
        module.main()

    self.assertIn("clrs.examples.run", str(ctx.exception))
    self._assert_deprecation_warning(caught, "__main__")


if __name__ == "__main__":
  unittest.main()
