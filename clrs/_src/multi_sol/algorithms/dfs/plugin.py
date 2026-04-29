"""DFS plugin wiring for modular multi-solution evaluation."""

from typing import Dict

import clrs
import jax
import numpy as np

from clrs._src.multi_sol.data.adapters import concat_tree
from clrs._src.multi_sol.evaluation import distribution_validation
from clrs._src.multi_sol.evaluation import reporting
from clrs._src.multi_sol.evaluation import reports
from clrs._src.multi_sol.evaluation.runners import evaluate_sampling_pair
from clrs._src.multi_sol.sampling import dfs as dfs_sampling
from clrs._src.multi_sol.validation import dfs as dfs_validation


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
  processed_samples = 0
  pred_batches = []
  outputs = []
  adjacency_batches = []

  while processed_samples < sample_count:
    feedback = next(sampler)
    batch_size = feedback.outputs[0].data.shape[0]
    outputs.append(feedback.outputs)
    new_rng_key, rng_key = jax.random.split(rng_key)
    cur_preds, _ = predict_fn(new_rng_key, feedback.features)
    pred_batches.append(cur_preds)
    processed_samples += batch_size
    adjacency_batches.append(feedback[0][0][1].data)

  outputs = concat_tree(outputs, axis=0)
  adjacency = concat_tree(adjacency_batches, axis=0)
  preds = concat_tree(pred_batches, axis=0)
  source_nodes = [0] * len(adjacency)
  out = clrs.evaluate(outputs, preds)

  def validate_fn(adjacency_item, parent_tree, _source):
    return dfs_validation.check_valid_dfsTree(adjacency_item, parent_tree)

  random_sampling = evaluate_sampling_pair(
      model_sample_fn=lambda data: dfs_sampling.sample_random_list(data),
      true_sample_fn=lambda data: dfs_sampling.sample_random_list(data),
      model_input=[preds],
      true_input=outputs,
      adjacency=adjacency,
      source_nodes=source_nodes,
      validate_fn=validate_fn,
  )
  argmax_sampling = evaluate_sampling_pair(
      model_sample_fn=lambda data: dfs_sampling.sample_argmax_listofdict(data),
      true_sample_fn=lambda data: dfs_sampling.sample_argmax_listofdatapoint(data),
      model_input=[preds],
      true_input=outputs,
      adjacency=adjacency,
      source_nodes=source_nodes,
      validate_fn=validate_fn,
  )
  upwards_sampling = evaluate_sampling_pair(
      model_sample_fn=lambda data: dfs_sampling.sample_upwards(data),
      true_sample_fn=lambda data: dfs_sampling.sample_upwards(data),
      model_input=[preds],
      true_input=outputs,
      adjacency=adjacency,
      source_nodes=source_nodes,
      validate_fn=validate_fn,
  )
  alt_upwards_sampling = evaluate_sampling_pair(
      model_sample_fn=lambda data: dfs_sampling.sample_altUpwards(data),
      true_sample_fn=lambda data: dfs_sampling.sample_altUpwards(data),
      model_input=[preds],
      true_input=outputs,
      adjacency=adjacency,
      source_nodes=source_nodes,
      validate_fn=validate_fn,
  )

  result_dict = reports.build_dfs_result_dict(
      adjacency_flat=[i.flatten() for i in adjacency],
      argmax=argmax_sampling,
      random_sampling=random_sampling,
      upwards=upwards_sampling,
      alt_upwards=alt_upwards_sampling,
  )
  sampling_methods = {
      "Argmax": lambda data: dfs_sampling.sample_argmax_listofdict(data),
      "Random": lambda data: dfs_sampling.sample_random_list(data),
      "Upwards": lambda data: dfs_sampling.sample_upwards(data),
      "altUpwards": lambda data: dfs_sampling.sample_altUpwards(data),
  }
  true_sampling_methods = {
      "Argmax": lambda data: dfs_sampling.sample_argmax_listofdatapoint(data),
      "Random": lambda data: dfs_sampling.sample_random_list(data),
      "Upwards": lambda data: dfs_sampling.sample_upwards(data),
      "altUpwards": lambda data: dfs_sampling.sample_altUpwards(data),
  }
  sampling_summary = distribution_validation.evaluate_mixed_sampling_methods(
      model_methods=sampling_methods,
      true_methods=true_sampling_methods,
      model_input=[preds],
      true_input=outputs,
      adjacency=adjacency,
      source_nodes=source_nodes,
      validate_fn=validate_fn,
      n_samples=NSE,
      curve_max_graphs=curve_max_graphs,
  )
  result_dict.update(sampling_summary["result_dict"])
  if vd_flag:
    algorithm_rng = np.random.default_rng()
    algorithm_summary = distribution_validation.evaluate_sampling_sources(
        sources={
            ("DFS", "Algorithm"): (
                lambda _data: _sample_randomized_dfs_algorithm(
                    adjacency, algorithm_rng),
                None,
            ),
        },
        adjacency=adjacency,
        source_nodes=source_nodes,
        validate_fn=validate_fn,
        n_samples=NSE,
        curve_max_graphs=curve_max_graphs,
    )
    result_dict.update(algorithm_summary["result_dict"])
    sampling_summary["curves"].extend(algorithm_summary["curves"])
    out.update(algorithm_summary["scalar_metrics"])
    distribution_validation.save_sampling_curve_artifacts(
        sampling_summary["curves"],
        filename=f"{filename}_DFS",
        output_dir=output_dir,
    )
  report_sink = save_results_fn or reporting.discard_report
  report_sink(result_dict, f"{filename}_DFS")

  out.update(sampling_summary["scalar_metrics"])
  if extras:
    out.update(extras)
  return {k: _unpack(v) for k, v in out.items()}


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


def _unpack(v):
  try:
    return v.item()
  except (AttributeError, ValueError):
    return v
