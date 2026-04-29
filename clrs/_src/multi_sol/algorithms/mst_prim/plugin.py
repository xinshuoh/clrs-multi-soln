"""MST-Prim plugin wiring for modular multi-solution evaluation."""

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
from clrs._src.multi_sol.sampling import mst_prim as mst_sampling
from clrs._src.multi_sol.validation import mst_prim as mst_validation


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
  processed_samples = 0
  pred_batches = []
  outputs = []
  adjacency_batches = []
  source_batches = []

  while processed_samples < sample_count:
    feedback = next(sampler)
    batch_size = feedback.outputs[0].data.shape[0]
    outputs.append(feedback.outputs)
    new_rng_key, rng_key = jax.random.split(rng_key)
    cur_preds, _ = predict_fn(new_rng_key, feedback.features)
    pred_batches.append(cur_preds)
    processed_samples += batch_size
    adjacency_batches.append(feedback[0][0][2].data)
    source_batches.append(np.argmax(feedback[0][0][1].data, axis=1))

  outputs = concat_tree(outputs, axis=0)
  adjacency = concat_tree(adjacency_batches, axis=0)
  source_nodes = concat_tree(source_batches, axis=0).astype(int)
  preds = concat_tree(pred_batches, axis=0)
  out = clrs.evaluate(outputs, preds)

  random_sampling = evaluate_sampling_pair(
      model_sample_fn=lambda data: dfs_sampling.sample_random_list(data),
      true_sample_fn=lambda data: dfs_sampling.sample_random_list(data),
      model_input=[preds],
      true_input=outputs,
      adjacency=adjacency,
      source_nodes=source_nodes,
      validate_fn=mst_validation.check_valid_mst_prim_tree,
  )
  argmax_sampling = evaluate_sampling_pair(
      model_sample_fn=lambda data: dfs_sampling.sample_argmax_listofdict(data),
      true_sample_fn=lambda data: dfs_sampling.sample_argmax_listofdatapoint(data),
      model_input=[preds],
      true_input=outputs,
      adjacency=adjacency,
      source_nodes=source_nodes,
      validate_fn=mst_validation.check_valid_mst_prim_tree,
  )
  tree_sampling = evaluate_sampling_pair(
      model_sample_fn=lambda data, s: mst_sampling.sample_mst_prim_tree(
          adjacency, s, data),
      true_sample_fn=lambda data, s: mst_sampling.sample_mst_prim_tree(
          adjacency, s, data),
      model_input=[preds],
      true_input=outputs,
      adjacency=adjacency,
      source_nodes=source_nodes,
      validate_fn=mst_validation.check_valid_mst_prim_tree,
      s=source_nodes,
  )
  greedy_sampling = evaluate_sampling_pair(
      model_sample_fn=lambda data, s: mst_sampling.sample_mst_prim_greedy(
          adjacency, s, data),
      true_sample_fn=lambda data, s: mst_sampling.sample_mst_prim_greedy(
          adjacency, s, data),
      model_input=[preds],
      true_input=outputs,
      adjacency=adjacency,
      source_nodes=source_nodes,
      validate_fn=mst_validation.check_valid_mst_prim_tree,
      s=source_nodes,
  )

  result_dict = reports.build_mst_prim_result_dict(
      adjacency_flat=[i.flatten() for i in adjacency],
      argmax=argmax_sampling,
      random_sampling=random_sampling,
      tree=tree_sampling,
      greedy=greedy_sampling,
  )
  model_methods = {
      "Argmax": lambda data: dfs_sampling.sample_argmax_listofdict(data),
      "Random": lambda data: dfs_sampling.sample_random_list(data),
      "Tree": lambda data: mst_sampling.sample_mst_prim_tree(
          adjacency, source_nodes, data),
      "Greedy": lambda data: mst_sampling.sample_mst_prim_greedy(
          adjacency, source_nodes, data),
  }
  true_methods = {
      "Argmax": lambda data: dfs_sampling.sample_argmax_listofdatapoint(data),
      "Random": lambda data: dfs_sampling.sample_random_list(data),
      "Tree": lambda data: mst_sampling.sample_mst_prim_tree(
          adjacency, source_nodes, data),
      "Greedy": lambda data: mst_sampling.sample_mst_prim_greedy(
          adjacency, source_nodes, data),
  }
  sampling_summary = distribution_validation.evaluate_mixed_sampling_methods(
      model_methods=model_methods,
      true_methods=true_methods,
      model_input=[preds],
      true_input=outputs,
      adjacency=adjacency,
      source_nodes=source_nodes,
      validate_fn=mst_validation.check_valid_mst_prim_tree,
      n_samples=NSE,
      curve_max_graphs=curve_max_graphs,
  )
  result_dict.update(sampling_summary["result_dict"])
  if vd_flag:
    algorithm_rng = np.random.default_rng(np.random.randint(0, 2**32))
    algorithm_summary = distribution_validation.evaluate_sampling_sources(
        sources={
            ("Prim", "Algorithm"): (
                lambda _data: _sample_randomized_prim_algorithm(
                    adjacency, source_nodes, algorithm_rng),
                None,
            ),
        },
        adjacency=adjacency,
        source_nodes=source_nodes,
        validate_fn=mst_validation.check_valid_mst_prim_tree,
        n_samples=NSE,
        curve_max_graphs=curve_max_graphs,
    )
    result_dict.update(algorithm_summary["result_dict"])
    sampling_summary["curves"].extend(algorithm_summary["curves"])
    out.update(algorithm_summary["scalar_metrics"])
    distribution_validation.save_sampling_curve_artifacts(
        sampling_summary["curves"],
        filename=filename,
        output_dir=output_dir,
    )
  report_sink = save_results_fn or reporting.discard_report
  report_sink(result_dict, filename)

  out.update(sampling_summary["scalar_metrics"])
  if extras:
    out.update(extras)
  return {k: _unpack(v) for k, v in out.items()}


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


def _unpack(v):
  try:
    return v.item()
  except (AttributeError, ValueError):
    return v
