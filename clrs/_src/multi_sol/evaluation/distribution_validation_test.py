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


class DistributionValidationTest(unittest.TestCase):

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

  def _load_module_with_stubs(self):
    self._install_package("clrs")
    self._install_package("clrs._src")
    self._install_package("clrs._src.multi_sol")
    self._install_package("clrs._src.multi_sol.data")
    self._install_package("clrs._src.multi_sol.evaluation")

    self.calls = {}

    data_gen_module = types.ModuleType(
        "clrs._src.multi_sol.data.distribution_generation")
    def _build_bf_payload(**kwargs):
      self.calls["bf_payload"] = kwargs
      return types.SimpleNamespace(graph_size=3)

    def _build_dfs_payload(**kwargs):
      self.calls["dfs_payload"] = kwargs
      return types.SimpleNamespace(graph_size=4)

    def _generate_validation_dataframes(**kwargs):
      self.calls["gen_kwargs"] = kwargs
      return ["u"], ["e"]

    data_gen_module.build_bf_validation_payload = _build_bf_payload
    data_gen_module.build_dfs_validation_payload = _build_dfs_payload
    data_gen_module.generate_validation_dataframes = (
        _generate_validation_dataframes)
    self._install_module("clrs._src.multi_sol.data.distribution_generation",
                         data_gen_module)

    validate_module = types.ModuleType("clrs._src.validate_distributions")
    def _plot_bf_unique(dataframes, graph_size, output_dir="."):
      self.calls["plot_bf_unique"] = (dataframes, graph_size, output_dir)

    def _plot_bf_reuse(df, graph_size, output_dir="."):
      self.calls["plot_bf_reuse"] = (df, graph_size, output_dir)

    def _plot_bf_line(df, graph_size, output_dir="."):
      self.calls["plot_bf_line"] = (df, graph_size, output_dir)

    def _plot_dfs_unique(dataframes, graph_size, output_dir="."):
      self.calls["plot_dfs_unique"] = (dataframes, graph_size, output_dir)

    def _plot_dfs_reuse(df, graph_size, output_dir="."):
      self.calls["plot_dfs_reuse"] = (df, graph_size, output_dir)

    def _plot_dfs_line(df, graph_size, output_dir="."):
      self.calls["plot_dfs_line"] = (df, graph_size, output_dir)

    validate_module.plot_n_unique_by_n_extracted = _plot_bf_unique
    validate_module.plot_edge_reuse_matrix_list_mean = _plot_bf_reuse
    validate_module.line_plot = _plot_bf_line
    validate_module.plot_n_unique_by_n_extracted_dfs = _plot_dfs_unique
    validate_module.plot_edge_reuse_matrix_list_mean_dfs = _plot_dfs_reuse
    validate_module.line_plot_dfs = _plot_dfs_line
    self._install_module("clrs._src.validate_distributions", validate_module)

    module_path = (
        self._repo_root / "clrs" / "_src" / "multi_sol" / "evaluation" /
        "distribution_validation.py")
    spec = importlib.util.spec_from_file_location(
        "clrs._src.multi_sol.evaluation.distribution_validation", str(module_path))
    module = importlib.util.module_from_spec(spec)
    self._install_module("clrs._src.multi_sol.evaluation.distribution_validation",
                         module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

  def test_run_bf_distribution_validation(self):
    module = self._load_module_with_stubs()
    module.run_bf_distribution_validation(
        adjacency=np.ones((2, 3, 3)),
        source_nodes=np.asarray([0, 1]),
        outputs={"o": 1},
        preds={"p": 2},
        nse=55,
        output_dir="results/run-1",
    )

    self.assertEqual(self.calls["bf_payload"]["outputs"], {"o": 1})
    self.assertEqual(self.calls["bf_payload"]["preds"], {"p": 2})
    self.assertEqual(self.calls["gen_kwargs"]["nse"], 55)
    self.assertEqual(self.calls["gen_kwargs"]["mode"], "BF")
    self.assertEqual(self.calls["plot_bf_unique"], (["u"], 3, "results/run-1"))
    self.assertEqual(self.calls["plot_bf_reuse"], (["e"], 3, "results/run-1"))
    self.assertEqual(self.calls["plot_bf_line"], (["e"], 3, "results/run-1"))

  def test_run_dfs_distribution_validation(self):
    module = self._load_module_with_stubs()
    module.run_dfs_distribution_validation(
        adjacency=np.ones((2, 4, 4)),
        outputs={"o": 1},
        pred_batches=[{"p": 2}],
        nse=44,
        output_dir="results/run-2",
    )

    self.assertEqual(self.calls["dfs_payload"]["outputs"], {"o": 1})
    self.assertEqual(self.calls["dfs_payload"]["preds"], [{"p": 2}])
    self.assertEqual(self.calls["gen_kwargs"]["nse"], 44)
    self.assertEqual(self.calls["gen_kwargs"]["mode"], "DFS")
    self.assertEqual(self.calls["plot_dfs_unique"], (["u"], 4, "results/run-2"))
    self.assertEqual(self.calls["plot_dfs_reuse"], (["e"], 4, "results/run-2"))
    self.assertEqual(self.calls["plot_dfs_line"], (["e"], 4, "results/run-2"))

  def test_sampling_sources_include_per_sample_rows_and_graph_limit(self):
    module = self._load_module_with_stubs()
    adjacency = np.ones((2, 3, 3))
    source_nodes = np.asarray([0, 0])
    calls = []

    def _sample(_unused):
      calls.append(1)
      if len(calls) == 1:
        return np.asarray([[0, 0, 1], [0, 1, 1]])
      return np.asarray([[0, 0, 2], [0, 2, 1]])

    def _validate(_adjacency, tree, _source):
      return int(tree[0]) == 0

    out = module.evaluate_sampling_sources(
        sources={("BFS", "Algorithm"): (_sample, None)},
        adjacency=adjacency,
        source_nodes=source_nodes,
        validate_fn=_validate,
        n_samples=2,
        curve_max_graphs=1,
    )

    self.assertEqual(len(out["curves"]), 2)
    self.assertEqual(out["curves"][0]["Graph"], 0)
    self.assertEqual(out["curves"][0]["Sample_Valid"], True)
    self.assertEqual(out["curves"][0]["Sample_Unique"], True)
    self.assertEqual(out["curves"][0]["Sample_Valid_Unique"], True)
    self.assertEqual(out["curves"][0]["Solution_Key"], "0|0|1")
    self.assertEqual(out["curves"][0]["Parent_Tree"], [0, 0, 1])
    self.assertEqual(out["curves"][1]["Cumulative_Valid_Unique"], 2)
    self.assertIn("BFS_Algorithm_Valid_Unique", out["scalar_metrics"])


if __name__ == "__main__":
  unittest.main()
