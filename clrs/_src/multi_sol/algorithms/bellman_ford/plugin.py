"""Bellman-Ford plugin wiring for modular multi-solution evaluation."""

from typing import Dict

import numpy as np

from clrs._src.multi_sol.data import adapters
from clrs._src.multi_sol.evaluation import batch_evaluation
from clrs._src.multi_sol.sampling import bellman_ford as bf_sampling
from clrs._src.multi_sol.sampling import dfs as dfs_sampling
from clrs._src.multi_sol.validation import bellman_ford as bf_validation

distribution_validation = batch_evaluation.distribution_validation


def evaluate_bf_multisol_batch(
    sampler,
    predict_fn,
    sample_count,
    rng_key,
    extras,
    save_results_fn=None,
    filename="bf_accuracy",
    vd_flag=False,
    NSE=100,
    output_dir=".",
    curve_max_graphs=None,
) -> Dict[str, float]:
  """Collect, evaluate, sample, validate and save Bellman-Ford results."""
  return batch_evaluation.evaluate_multisol_batch(
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      extras=extras,
      batch_extractor=adapters.extract_bellman_ford_graph_and_source,
      validate_fn=bf_validation.check_valid_bf_paths,
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
              "Beam",
              lambda data, batch: bf_sampling.sample_beamsearch(
                  batch.adjacency, batch.source_nodes, data),
              lambda data, batch: bf_sampling.sample_beamsearch(
                  batch.adjacency, batch.source_nodes, data),
          ),
          batch_evaluation.SamplingMethod(
              "Greedy",
              lambda data, batch: bf_sampling.sample_greedysearch(
                  batch.adjacency, batch.source_nodes, data),
              lambda data, batch: bf_sampling.sample_greedysearch(
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
          "BellmanFord",
          "Algorithm",
          lambda batch, rng: _sample_randomized_bellman_ford_algorithm(
              batch.adjacency, batch.source_nodes, rng),
      ),
  )


def _sample_randomized_bellman_ford_algorithm(adjacency, source_nodes, rng):
  return [
      _randomized_bellman_ford_tree(np.asarray(graph), int(source), rng)
      for graph, source in zip(adjacency, source_nodes)
  ]


def _randomized_bellman_ford_tree(adjacency, source, rng):
  n = adjacency.shape[0]
  d = np.zeros(n)
  pi = np.arange(n, dtype=int)
  msk = np.zeros(n)
  d[source] = 0
  msk[source] = 1

  shuffled_sources = rng.permutation(n)
  shuffled_targets = rng.permutation(n)
  while True:
    prev_d = np.copy(d)
    prev_msk = np.copy(msk)
    for u in shuffled_sources:
      for v in shuffled_targets:
        if prev_msk[u] == 1 and adjacency[u, v] != 0:
          if msk[v] == 0 or prev_d[u] + adjacency[u, v] < d[v]:
            d[v] = prev_d[u] + adjacency[u, v]
            pi[v] = int(u)
          msk[v] = 1
    if np.all(d == prev_d):
      break
  return pi
