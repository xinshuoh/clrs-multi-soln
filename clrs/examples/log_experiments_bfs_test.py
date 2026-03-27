import importlib.util
import pathlib
import sys
import types
import unittest


_MISSING = object()


def _find_repo_root() -> pathlib.Path:
  cur = pathlib.Path(__file__).resolve()
  for parent in [cur] + list(cur.parents):
    if (parent / "setup.py").exists():
      return parent
  raise RuntimeError("Could not locate repository root from test path.")


class BfsCollectAndEvalWrapperTest(unittest.TestCase):

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

  def _load_log_experiments(self, plugin_fn):
    self._install_package("clrs")
    self._install_package("clrs._src")
    self._install_package("clrs._src.algorithms")
    self._install_package("clrs._src.multi_sol")
    self._install_package("clrs._src.multi_sol.algorithms")
    self._install_package("clrs._src.multi_sol.algorithms.bfs")
    self._install_package("clrs._src.multi_sol.data")
    self._install_package("clrs._src.multi_sol.sampling")
    self._install_package("clrs._src.multi_sol.algorithms.dfs")
    self._install_package("clrs._src.multi_sol.algorithms.bellman_ford")

    pandas_module = types.ModuleType("pandas")
    pandas_module.set_option = lambda *args, **kwargs: None
    pandas_module.DataFrame = type("DataFrame", (), {"from_dict": staticmethod(dict)})
    self._install_module("pandas", pandas_module)

    jax_module = types.ModuleType("jax")
    jax_module.tree_util = types.SimpleNamespace(tree_map=lambda f, *x: f(*x))
    self._install_module("jax", jax_module)

    self._install_module("clrs._src.dfs_sampling",
                         types.ModuleType("clrs._src.dfs_sampling"))
    self._install_module("clrs._src.dfs_uniqueness_check",
                         types.ModuleType("clrs._src.dfs_uniqueness_check"))
    dfs_sampling_module = types.ModuleType("clrs._src.multi_sol.sampling.dfs")
    dfs_sampling_module.sample_random_list = lambda data: []
    dfs_sampling_module.sample_argmax = lambda data: []
    dfs_sampling_module.sample_argmax_listofdict = lambda data: []
    dfs_sampling_module.sample_argmax_listofdatapoint = lambda data: []
    dfs_sampling_module.sample_upwards = lambda data: []
    dfs_sampling_module.sample_altUpwards = lambda data: []
    self._install_module("clrs._src.multi_sol.sampling.dfs", dfs_sampling_module)

    bf_sampling_module = types.ModuleType("clrs._src.multi_sol.sampling.bellman_ford")
    bf_sampling_module.sample_beamsearch = lambda *args, **kwargs: []
    bf_sampling_module.sample_greedysearch = lambda *args, **kwargs: []
    self._install_module("clrs._src.multi_sol.sampling.bellman_ford",
                         bf_sampling_module)

    check_graphs_module = types.ModuleType("clrs._src.algorithms.check_graphs")
    dfs_verify_module = types.ModuleType(
        "clrs._src.algorithms.dfs_verification_tester")
    self._install_module("clrs._src.algorithms.check_graphs", check_graphs_module)
    self._install_module("clrs._src.algorithms.dfs_verification_tester",
                         dfs_verify_module)

    bf_beamsearch_module = types.ModuleType("clrs._src.algorithms.BF_beamsearch")
    bf_beamsearch_module.sample_beamsearch = lambda *args, **kwargs: None
    bf_beamsearch_module.sample_greedysearch = lambda *args, **kwargs: None
    self._install_module("clrs._src.algorithms.BF_beamsearch", bf_beamsearch_module)

    bf_uniqueness_module = types.ModuleType("clrs._src.bf_uniqueness_check")
    bf_uniqueness_module.check_uniqueness_bf = lambda *args, **kwargs: None
    self._install_module("clrs._src.bf_uniqueness_check", bf_uniqueness_module)

    validate_module = types.ModuleType("clrs._src.validate_distributions")
    validate_module.plot_edge_reuse_matrix_list_mean = lambda *args, **kwargs: None
    validate_module.plot_edge_reuse_matrix_list_mean_dfs = (
        lambda *args, **kwargs: None)
    validate_module.plot_n_unique_by_n_extracted = lambda *args, **kwargs: None
    validate_module.plot_n_unique_by_n_extracted_dfs = (
        lambda *args, **kwargs: None)
    validate_module.line_plot = lambda *args, **kwargs: None
    validate_module.line_plot_dfs = lambda *args, **kwargs: None
    self._install_module("clrs._src.validate_distributions", validate_module)

    distribution_generation = types.ModuleType(
        "clrs._src.multi_sol.data.distribution_generation")
    distribution_generation.build_bf_validation_payload = (
        lambda *args, **kwargs: None)
    distribution_generation.build_dfs_validation_payload = (
        lambda *args, **kwargs: None)
    distribution_generation.generate_validation_dataframes = (
        lambda *args, **kwargs: ([], [], []))
    self._install_module("clrs._src.multi_sol.data.distribution_generation",
                         distribution_generation)

    plugin_module = types.ModuleType("clrs._src.multi_sol.algorithms.bfs.plugin")
    plugin_module.evaluate_bfs_multisol_batch = plugin_fn
    self._install_module("clrs._src.multi_sol.algorithms.bfs.plugin", plugin_module)
    dfs_plugin_module = types.ModuleType("clrs._src.multi_sol.algorithms.dfs.plugin")
    dfs_plugin_module.evaluate_dfs_multisol_batch = lambda **kwargs: {"delegated_dfs": 1.0}
    self._install_module("clrs._src.multi_sol.algorithms.dfs.plugin", dfs_plugin_module)
    bf_plugin_module = types.ModuleType(
        "clrs._src.multi_sol.algorithms.bellman_ford.plugin")
    bf_plugin_module.evaluate_bf_multisol_batch = lambda **kwargs: {"delegated_bf": 1.0}
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

  def test_bfs_multi_collect_and_eval_delegates_to_plugin(self):
    captured = {}

    def plugin_fn(**kwargs):
      captured.update(kwargs)
      return {"delegated": 1.0}

    module = self._load_log_experiments(plugin_fn)
    sampler = object()
    predict_fn = object()
    extras = {"extra": "value"}
    result = module.BFS_multi_collect_and_eval(
        sampler=sampler,
        predict_fn=predict_fn,
        sample_count=7,
        rng_key=123,
        extras=extras,
        filename="regression",
        vd_flag=True,
        NSE=11,
    )

    self.assertEqual(result, {"delegated": 1.0})
    self.assertIs(captured["sampler"], sampler)
    self.assertIs(captured["predict_fn"], predict_fn)
    self.assertEqual(captured["sample_count"], 7)
    self.assertEqual(captured["rng_key"], 123)
    self.assertEqual(captured["extras"], extras)
    self.assertEqual(captured["filename"], "regression")
    self.assertIs(captured["save_results_fn"], module.save_results)


if __name__ == "__main__":
  unittest.main()
