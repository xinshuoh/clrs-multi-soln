"""MST-Prim plugin wiring for modular multi-solution evaluation."""

from typing import Dict

import numpy as np

from clrs._src.multi_sol.data import adapters
from clrs._src.multi_sol.evaluation import batch_evaluation
from clrs._src.multi_sol.sampling import dfs as dfs_sampling
from clrs._src.multi_sol.sampling import mst_prim as mst_sampling
from clrs._src.multi_sol.validation import mst_prim as mst_validation

distribution_validation = batch_evaluation.distribution_validation


def evaluate_mst_prim_multisol_batch(
    sampler,
    predict_fn,
    sample_count,
    rng_key,
    extras,
    save_results_fn=None,
    filename="mst_prim_accuracy",
    vd_flag=False,
    NSE=100,
    output_dir=".",
    curve_max_graphs=None,
) -> Dict[str, float]:
  """Collect, evaluate, sample, validate and save MST-Prim results."""
  return batch_evaluation.evaluate_multisol_batch(
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      extras=extras,
      batch_extractor=adapters.extract_mst_prim_graph_and_source,
      validate_fn=mst_validation.check_valid_mst_prim_tree,
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
              "Tree",
              lambda data, batch: mst_sampling.sample_mst_prim_tree(
                  batch.adjacency, batch.source_nodes, data),
              lambda data, batch: mst_sampling.sample_mst_prim_tree(
                  batch.adjacency, batch.source_nodes, data),
          ),
          batch_evaluation.SamplingMethod(
              "Greedy",
              lambda data, batch: mst_sampling.sample_mst_prim_greedy(
                  batch.adjacency, batch.source_nodes, data),
              lambda data, batch: mst_sampling.sample_mst_prim_greedy(
                  batch.adjacency, batch.source_nodes, data),
          ),
      ),
      save_results_fn=save_results_fn,
      filename=filename,
      vd_flag=vd_flag,
      n_samples=NSE,
      output_dir=output_dir,
      curve_max_graphs=curve_max_graphs,
      algorithm_source=batch_evaluation.AlgorithmSamplingSource(
          "Prim",
          "Algorithm",
          lambda batch, rng: _sample_randomized_prim_algorithm(
              batch.adjacency, batch.source_nodes, rng),
      ),
  )


def _sample_randomized_prim_algorithm(adjacency, source_nodes, rng):
  return [
      _randomized_prim_tree(np.asarray(graph), int(source), rng)
      for graph, source in zip(adjacency, source_nodes)
  ]


def _randomized_prim_tree(adjacency, source, rng):
  n = adjacency.shape[0]
  key = np.zeros(n)
  mark = np.zeros(n)
  in_queue = np.zeros(n)
  pi = np.arange(n, dtype=int)
  key[source] = 0
  in_queue[source] = 1

  for _ in range(n):
    effective_keys = np.where(in_queue == 1, key, np.inf)
    min_key_val = np.min(effective_keys)
    if np.isinf(min_key_val):
      break
    candidates = np.where(effective_keys == min_key_val)[0]
    u = int(rng.choice(candidates))
    if in_queue[u] == 0:
      break
    mark[u] = 1
    in_queue[u] = 0
    for v in range(n):
      if adjacency[u, v] != 0:
        if mark[v] == 0 and (in_queue[v] == 0 or adjacency[u, v] < key[v]):
          pi[v] = u
          key[v] = adjacency[u, v]
          in_queue[v] = 1
  return pi
