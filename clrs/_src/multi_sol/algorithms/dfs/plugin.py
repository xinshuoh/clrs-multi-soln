"""DFS plugin wiring for modular multi-solution evaluation."""

from typing import Dict

import numpy as np

from clrs._src.multi_sol.data import adapters
from clrs._src.multi_sol.evaluation import batch_evaluation
from clrs._src.multi_sol.sampling import dfs as dfs_sampling
from clrs._src.multi_sol.validation import dfs as dfs_validation

distribution_validation = batch_evaluation.distribution_validation


def evaluate_dfs_multisol_batch(
    sampler,
    predict_fn,
    sample_count,
    rng_key,
    extras,
    save_results_fn=None,
    filename="dfs_accuracy",
    vd_flag=False,
    NSE=100,
    output_dir=".",
    curve_max_graphs=None,
) -> Dict[str, float]:
  """Collect, evaluate, sample, validate and save DFS multi-solution results."""
  return batch_evaluation.evaluate_multisol_batch(
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      extras=extras,
      batch_extractor=adapters.extract_dfs_graph_and_source,
      validate_fn=_validate_dfs_tree,
      sampling_methods=(
          batch_evaluation.SamplingMethod(
              "Argmax",
              lambda data, _batch: dfs_sampling.sample_argmax_listofdict(data),
              lambda data, _batch: dfs_sampling.sample_argmax_listofdatapoint(
                  data),
          ),
          batch_evaluation.SamplingMethod(
              "Random",
              lambda data, _batch: dfs_sampling.sample_random_list(data),
              lambda data, _batch: dfs_sampling.sample_random_list(data),
          ),
          batch_evaluation.SamplingMethod(
              "Upwards",
              lambda data, _batch: dfs_sampling.sample_upwards(data),
              lambda data, _batch: dfs_sampling.sample_upwards(data),
          ),
          batch_evaluation.SamplingMethod(
              "altUpwards",
              lambda data, _batch: dfs_sampling.sample_altUpwards(data),
              lambda data, _batch: dfs_sampling.sample_altUpwards(data),
          ),
      ),
      save_results_fn=save_results_fn,
      filename=filename,
      vd_flag=vd_flag,
      n_samples=NSE,
      output_dir=output_dir,
      curve_max_graphs=curve_max_graphs,
      algorithm_source=batch_evaluation.AlgorithmSamplingSource(
          "DFS",
          "Algorithm",
          lambda batch, rng: _sample_randomized_dfs_algorithm(
              batch.adjacency, rng),
      ),
  )


def _sample_randomized_dfs_algorithm(adjacency, rng):
  return [
      _randomized_dfs_tree(np.asarray(graph), rng)
      for graph in adjacency
  ]


def _randomized_dfs_tree(adjacency, rng):
  n = adjacency.shape[0]
  color = np.zeros(n, dtype=np.int32)
  pi = np.arange(n, dtype=int)
  s_prev = np.arange(n, dtype=int)
  shuffled = rng.permutation(n)

  for s in range(n):
    if color[s] != 0:
      continue
    s_last = s
    u = s
    while True:
      if color[u] == 0:
        color[u] = 1

      for v in shuffled:
        if adjacency[u, v] != 0 and color[v] == 0:
          pi[v] = u
          color[v] = 1
          s_prev[v] = s_last
          s_last = v
          break

      if s_last == u:
        color[u] = 2
        if s_prev[u] == u:
          break
        parent = s_prev[s_last]
        s_prev[s_last] = s_last
        s_last = parent

      u = s_last
  return pi


def _validate_dfs_tree(adjacency, parent_tree, _source):
  return dfs_validation.check_valid_dfsTree(adjacency, parent_tree)
