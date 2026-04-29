import collections
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


def _extract_prob_matrices(outs_or_preds):
  matrices = []
  for item in outs_or_preds:
    distlist = item["pi"].data if isinstance(item, dict) else item.data
    matrices.extend([np.asarray(x) for x in distlist])
  return matrices


def _stub_bfs_sample_argmax(outs_or_preds, source_nodes=None):
  prob_matrix_list = _extract_prob_matrices(outs_or_preds)
  if source_nodes is None:
    source_nodes = [0] * len(prob_matrix_list)
  if isinstance(source_nodes, int):
    source_nodes = [source_nodes] * len(prob_matrix_list)
  trees = []
  for i, prob_matrix in enumerate(prob_matrix_list):
    tree = np.argmax(prob_matrix, axis=1).astype(int)
    source = int(source_nodes[i])
    tree[source] = source
    trees.append(tree)
  return trees


def _stub_random_trees(outs_or_preds):
  return [
      np.zeros(prob_matrix.shape[0], dtype=int)
      for prob_matrix in _extract_prob_matrices(outs_or_preds)
  ]


def _stub_concat_tree(items, axis):
  first = items[0]
  if isinstance(first, np.ndarray):
    return np.concatenate(items, axis=axis)
  if isinstance(first, dict):
    return {
        key: _stub_concat_tree([item[key] for item in items], axis)
        for key in first
    }
  if isinstance(first, list):
    return [
        _stub_concat_tree([item[i] for item in items], axis)
        for i in range(len(first))
    ]
  if hasattr(first, "data"):
    return DummyDataPoint(np.concatenate([item.data for item in items], axis=axis))
  raise TypeError(f"Unsupported concat type: {type(first)}")


Features = collections.namedtuple("Features", ["inputs", "hints", "lengths"])
Feedback = collections.namedtuple("Feedback", ["features", "outputs"])


class EvaluateBfsPluginTest(unittest.TestCase):

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

  def _load_module(self, name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    self._install_module(name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

  def _load_plugin_with_stubs(self):
    clrs_pkg = self._install_package("clrs")
    self._install_package("clrs._src")
    self._install_package("clrs._src.multi_sol")
    self._install_package("clrs._src.multi_sol.algorithms")
    self._install_package("clrs._src.multi_sol.algorithms.bfs")
    self._install_package("clrs._src.multi_sol.data")
    self._install_package("clrs._src.multi_sol.evaluation")
    self._install_package("clrs._src.multi_sol.sampling")
    self._install_package("clrs._src.multi_sol.validation")

    clrs_pkg.evaluate = lambda outputs, preds: {"score": np.array(0.75)}

    jax_module = types.ModuleType("jax")
    jax_module.random = types.SimpleNamespace(
        split=lambda key: (key + 1, key + 2))
    self._install_module("jax", jax_module)

    dfs_sampling_module = types.ModuleType("clrs._src.dfs_sampling")
    dfs_sampling_module.sample_random_list = _stub_random_trees
    self._install_module("clrs._src.dfs_sampling", dfs_sampling_module)

    adapters_module = types.ModuleType("clrs._src.multi_sol.data.adapters")
    adapters_module.concat_tree = _stub_concat_tree
    adapters_module.extract_bfs_graph_and_source = (
        lambda feedback: (feedback.features.inputs[2].data,
                          np.argmax(feedback.features.inputs[1].data, axis=1)))
    self._install_module("clrs._src.multi_sol.data.adapters", adapters_module)

    metrics_path = (
        self._repo_root / "clrs" / "_src" / "multi_sol" / "evaluation" /
        "metrics.py")
    self._load_module("clrs._src.multi_sol.evaluation.metrics", metrics_path)

    runners_path = (
        self._repo_root / "clrs" / "_src" / "multi_sol" / "evaluation" /
        "runners.py")
    self._load_module("clrs._src.multi_sol.evaluation.runners", runners_path)

    reports_path = (
        self._repo_root / "clrs" / "_src" / "multi_sol" / "evaluation" /
        "reports.py")
    self._load_module("clrs._src.multi_sol.evaluation.reports", reports_path)
    reporting_module = types.ModuleType("clrs._src.multi_sol.evaluation.reporting")
    reporting_module.discard_report = lambda *_args, **_kwargs: None
    self._install_module("clrs._src.multi_sol.evaluation.reporting",
                         reporting_module)

    bfs_sampling_module = types.ModuleType("clrs._src.multi_sol.sampling.bfs")
    bfs_sampling_module.sample_bfs_categorical = _stub_bfs_sample_argmax
    bfs_sampling_module.sample_bfs_prim = _stub_bfs_sample_argmax
    bfs_sampling_module.sample_bfs_beam = (
        lambda outs_or_preds, source_nodes, beam_width=3: _stub_bfs_sample_argmax(
            outs_or_preds, source_nodes)
    )
    self._install_module("clrs._src.multi_sol.sampling.bfs", bfs_sampling_module)
    dfs_sampling_module_2 = types.ModuleType("clrs._src.multi_sol.sampling.dfs")
    dfs_sampling_module_2.sample_random_list = _stub_random_trees
    self._install_module("clrs._src.multi_sol.sampling.dfs", dfs_sampling_module_2)

    bfs_validation_module = types.ModuleType("clrs._src.multi_sol.validation.bfs")
    bfs_validation_module.check_valid_bfs_tree = (
        lambda adjacency, parent_tree, source: (
            len(parent_tree) == adjacency.shape[0] and
            parent_tree[int(source)] == int(source) and
            np.all((np.asarray(parent_tree) >= 0) &
                   (np.asarray(parent_tree) < adjacency.shape[0]))
        )
    )
    self._install_module("clrs._src.multi_sol.validation.bfs",
                         bfs_validation_module)

    plugin_path = (
        self._repo_root / "clrs" / "_src" / "multi_sol" / "algorithms" /
        "bfs" / "plugin.py")
    return self._load_module("clrs._src.multi_sol.algorithms.bfs.plugin",
                             plugin_path)

  def test_evaluate_bfs_multisol_batch_smoke(self):
    plugin = self._load_plugin_with_stubs()

    adjacency = np.array([
        [[0, 1, 1], [1, 0, 0], [1, 0, 0]],
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    ])
    source_one_hot = np.array([[1, 0, 0], [0, 1, 0]])
    output_prob = np.array([
        [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.7, 0.3, 0.0]],
        [[0.2, 0.8, 0.0], [0.0, 1.0, 0.0], [0.1, 0.9, 0.0]],
    ])
    pred_prob = np.array([
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.9, 0.1, 0.0]],
        [[0.3, 0.7, 0.0], [0.0, 1.0, 0.0], [0.2, 0.8, 0.0]],
    ])

    feedback = Feedback(
        features=Features(
            inputs=[
                DummyDataPoint(np.zeros((2, 3))),
                DummyDataPoint(source_one_hot),
                DummyDataPoint(adjacency),
            ],
            hints=[],
            lengths=np.array([3, 3]),
        ),
        outputs=[DummyDataPoint(output_prob)],
    )

    class ConstantSampler:
      def __iter__(self):
        return self

      def __next__(self):
        return feedback

    def predict_fn(rng_key, features):
      del rng_key, features
      return {"pi": DummyDataPoint(pred_prob)}, None

    saved = {}

    def save_results_fn(result_dict, filename):
      saved["result_dict"] = result_dict
      saved["filename"] = filename

    out = plugin.evaluate_bfs_multisol_batch(
        sampler=iter(ConstantSampler()),
        predict_fn=predict_fn,
        sample_count=2,
        rng_key=5,
        extras={"tag": "ok"},
        save_results_fn=save_results_fn,
        filename="bfs_smoke",
    )

    self.assertEqual(saved["filename"], "bfs_smoke")
    self.assertIn("Categorical_Model_Accuracy", saved["result_dict"])
    self.assertIn("Beam_True_Accuracy", saved["result_dict"])
    self.assertEqual(out["score"], 0.75)
    self.assertEqual(out["tag"], "ok")

  def test_evaluate_bfs_multisol_batch_without_report_sink(self):
    plugin = self._load_plugin_with_stubs()

    adjacency = np.array([
        [[0, 1, 1], [1, 0, 0], [1, 0, 0]],
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    ])
    source_one_hot = np.array([[1, 0, 0], [0, 1, 0]])
    output_prob = np.array([
        [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.7, 0.3, 0.0]],
        [[0.2, 0.8, 0.0], [0.0, 1.0, 0.0], [0.1, 0.9, 0.0]],
    ])
    pred_prob = np.array([
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.9, 0.1, 0.0]],
        [[0.3, 0.7, 0.0], [0.0, 1.0, 0.0], [0.2, 0.8, 0.0]],
    ])

    feedback = Feedback(
        features=Features(
            inputs=[
                DummyDataPoint(np.zeros((2, 3))),
                DummyDataPoint(source_one_hot),
                DummyDataPoint(adjacency),
            ],
            hints=[],
            lengths=np.array([3, 3]),
        ),
        outputs=[DummyDataPoint(output_prob)],
    )

    class ConstantSampler:
      def __iter__(self):
        return self

      def __next__(self):
        return feedback

    def predict_fn(rng_key, features):
      del rng_key, features
      return {"pi": DummyDataPoint(pred_prob)}, None

    out = plugin.evaluate_bfs_multisol_batch(
        sampler=iter(ConstantSampler()),
        predict_fn=predict_fn,
        sample_count=2,
        rng_key=5,
        extras={"tag": "ok"},
        filename="bfs_smoke",
    )

    self.assertEqual(out["score"], 0.75)
    self.assertEqual(out["tag"], "ok")


if __name__ == "__main__":
  unittest.main()
