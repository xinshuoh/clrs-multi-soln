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


def _load_module(name: str, path: pathlib.Path):
  spec = importlib.util.spec_from_file_location(name, str(path))
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


class DummyDatapoint:
  def __init__(self, data):
    self.data = data


class BFSSamplingTest(unittest.TestCase):

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

  def _install_package(self, name):
    module = types.ModuleType(name)
    module.__path__ = []
    self._install_module(name, module)
    return module

  def _load_wrapper(self):
    self._install_package("clrs")
    self._install_package("clrs._src")
    self._install_package("clrs._src.multi_sol")
    self._install_package("clrs._src.multi_sol.sampling")

    base_module = types.ModuleType("clrs._src.multi_sol.sampling.base")

    def extract_prob_matrices(outs_or_preds):
      out = []
      for item in outs_or_preds:
        distlist = item["pi"].data if isinstance(item, dict) else item.data
        out.extend([np.asarray(x) for x in distlist])
      return out

    def normalize_rows(prob_matrix):
      normalized = prob_matrix.astype(np.float64, copy=True)
      row_sums = normalized.sum(axis=1)
      nonzero = row_sums > 0
      normalized[nonzero] = normalized[nonzero] / row_sums[nonzero][:, None]
      return normalized

    def as_index_list(values, expected_len):
      values = np.asarray(values)
      if values.ndim == 0:
        return [int(values)] * expected_len
      return values

    def sample_index(probabilities, fallback=None):
      probabilities = np.asarray(probabilities, dtype=np.float64)
      total = probabilities.sum()
      if total <= 0:
        if fallback == "uniform":
          return int(np.random.randint(len(probabilities)))
        return fallback
      return int(np.random.choice(len(probabilities), p=probabilities / total))

    base_module.extract_prob_matrices = extract_prob_matrices
    base_module.normalize_rows = normalize_rows
    base_module.as_index_list = as_index_list
    base_module.sample_index = sample_index
    self._install_module("clrs._src.multi_sol.sampling.base", base_module)

    bfs_module = _load_module(
        "clrs._src.multi_sol.sampling.bfs",
        self._repo_root / "clrs" / "_src" / "multi_sol" / "sampling" / "bfs.py",
    )
    self._install_module("clrs._src.multi_sol.sampling.bfs", bfs_module)

    wrapper_module = _load_module(
        "clrs._src.bfs_sampling",
        self._repo_root / "clrs" / "_src" / "bfs_sampling.py",
    )
    return wrapper_module, bfs_module

  def test_wrapper_exports_modular_bfs_api(self):
    bfs_sampling, modular_bfs = self._load_wrapper()
    expected_api = (
        "sample_bfs_prim",
        "prim_like_sampler",
        "sample_bfs_categorical",
        "sample_bfs_beam",
        "bfs_beam_sampler",
    )
    self.assertEqual(bfs_sampling.__all__, expected_api)
    for name in expected_api:
      self.assertIs(getattr(bfs_sampling, name), getattr(modular_bfs, name))

  def test_prim_like_sampler_simple_path(self):
    bfs_sampling, _ = self._load_wrapper()
    prob_matrix = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    trees = bfs_sampling.sample_bfs_prim([DummyDatapoint([prob_matrix])], [0])
    np.testing.assert_array_equal(trees[0], np.array([0, 0, 1]))

  def test_prim_like_sampler_branching(self):
    bfs_sampling, _ = self._load_wrapper()
    prob_matrix = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    trees = bfs_sampling.sample_bfs_prim([DummyDatapoint([prob_matrix])], [0])
    np.testing.assert_array_equal(trees[0], np.array([0, 0, 0]))

  def test_prim_like_sampler_disconnected_invariants(self):
    bfs_sampling, _ = self._load_wrapper()
    prob_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ])
    trees = bfs_sampling.sample_bfs_prim([DummyDatapoint([prob_matrix])], [0])
    self.assertEqual(trees[0][0], 0)
    self.assertTrue(np.all((trees[0] >= 0) & (trees[0] < 3)))

  def test_categorical_sampling_is_deterministic_for_one_hot_rows(self):
    bfs_sampling, _ = self._load_wrapper()
    prob_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    trees = bfs_sampling.sample_bfs_categorical([DummyDatapoint([prob_matrix])])
    np.testing.assert_array_equal(trees[0], np.array([0, 1, 2]))

  def test_beam_search_prefers_high_mass_parents(self):
    bfs_sampling, _ = self._load_wrapper()
    prob_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.95, 0.05, 0.0],
        [0.80, 0.20, 0.0],
    ])
    trees = bfs_sampling.sample_bfs_beam(
        [DummyDatapoint([prob_matrix])], [0], beam_width=2)
    np.testing.assert_array_equal(trees[0], np.array([0, 0, 0]))


if __name__ == "__main__":
  unittest.main()
