"""BFS plugin wiring for modular multi-solution evaluation."""

from typing import Dict

import numpy as np

from clrs._src.multi_sol.data import adapters
from clrs._src.multi_sol.evaluation import batch_evaluation
from clrs._src.multi_sol.sampling import bfs as bfs_sampling
from clrs._src.multi_sol.sampling import dfs as dfs_sampling
from clrs._src.multi_sol.validation import bfs as bfs_validation

distribution_validation = batch_evaluation.distribution_validation


def evaluate_bfs_multisol_batch(
    sampler,
    predict_fn,
    sample_count,
    rng_key,
    extras,
    save_results_fn=None,
    filename="bfs_accuracy",
    vd_flag=False,
    NSE=100,
    output_dir=".",
    curve_max_graphs=None,
) -> Dict[str, float]:
  """Collect, evaluate, sample, validate and save BFS multi-solution results."""
  return batch_evaluation.evaluate_multisol_batch(
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      extras=extras,
      batch_extractor=adapters.extract_bfs_graph_and_source,
      validate_fn=bfs_validation.check_valid_bfs_tree,
      sampling_methods=(
          batch_evaluation.SamplingMethod(
              "Categorical",
              lambda data, _batch: bfs_sampling.sample_bfs_categorical(data),
              lambda data, _batch: bfs_sampling.sample_bfs_categorical(data),
          ),
          batch_evaluation.SamplingMethod(
              "Random",
              lambda data, _batch: dfs_sampling.sample_random_list(data),
              lambda data, _batch: dfs_sampling.sample_random_list(data),
          ),
          batch_evaluation.SamplingMethod(
              "Prim",
              lambda data, batch: bfs_sampling.sample_bfs_prim(
                  data, batch.source_nodes),
              lambda data, batch: bfs_sampling.sample_bfs_prim(
                  data, batch.source_nodes),
          ),
          batch_evaluation.SamplingMethod(
              "Beam",
              lambda data, batch: bfs_sampling.sample_bfs_beam(
                  data, batch.source_nodes, beam_width=3),
              lambda data, batch: bfs_sampling.sample_bfs_beam(
                  data, batch.source_nodes, beam_width=3),
          ),
      ),
      save_results_fn=save_results_fn,
      filename=filename,
      vd_flag=vd_flag,
      n_samples=NSE,
      output_dir=output_dir,
      curve_max_graphs=curve_max_graphs,
      algorithm_source=batch_evaluation.AlgorithmSamplingSource(
          "BFS",
          "Algorithm",
          lambda batch, rng: _sample_randomized_bfs_algorithm(
              batch.adjacency, batch.source_nodes, rng),
      ),
      include_source_nodes=True,
  )


def _sample_randomized_bfs_algorithm(adjacency, source_nodes, rng):
  return [
      _randomized_bfs_tree(np.asarray(graph), int(source), rng)
      for graph, source in zip(adjacency, source_nodes)
  ]


def _randomized_bfs_tree(adjacency, source, rng):
  n = adjacency.shape[0]
  reach = np.zeros(n, dtype=bool)
  pi = np.arange(n, dtype=int)
  reach[source] = True

  while True:
    prev_reach = np.copy(reach)
    sources = np.where(prev_reach)[0]
    rng.shuffle(sources)
    for src in sources:
      for child in range(n):
        if adjacency[src, child] > 0:
          if pi[child] == child and child != source:
            pi[child] = int(src)
          reach[child] = True
    if np.all(reach == prev_reach):
      break
  return pi
