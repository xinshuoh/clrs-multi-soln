"""BFS plugin wiring for modular multi-solution evaluation."""

import copy
from typing import Dict

import clrs
import jax
import numpy as np

from clrs._src.multi_sol.data.adapters import concat_tree
from clrs._src.multi_sol.data.adapters import extract_bfs_graph_and_source
from clrs._src.multi_sol.evaluation import distribution_validation
from clrs._src.multi_sol.evaluation import reporting
from clrs._src.multi_sol.evaluation import reports
from clrs._src.multi_sol.evaluation.runners import evaluate_sampling_pair
from clrs._src.multi_sol.sampling import bfs as bfs_sampling
from clrs._src.multi_sol.sampling import dfs as dfs_sampling
from clrs._src.multi_sol.validation import bfs as bfs_validation


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
) -> Dict[str, float]:
  """Collect, evaluate, sample, validate and save BFS multi-solution results."""
  processed_samples = 0
  preds = []
  outputs = []
  adjacency_batches = []
  source_batches = []

  while processed_samples < sample_count:
    feedback = next(sampler)
    batch_size = feedback.outputs[0].data.shape[0]
    outputs.append(feedback.outputs)
    new_rng_key, rng_key = jax.random.split(rng_key)
    cur_preds, _ = predict_fn(new_rng_key, feedback.features)
    preds.append(cur_preds)
    processed_samples += batch_size
    adjacency, source = extract_bfs_graph_and_source(feedback)
    adjacency_batches.append(adjacency)
    source_batches.append(source)

  outputs = concat_tree(outputs, axis=0)
  adjacency = concat_tree(adjacency_batches, axis=0)
  source_nodes = concat_tree(source_batches, axis=0).astype(int)
  preds = concat_tree(preds, axis=0)
  out = clrs.evaluate(outputs, preds)

  categorical = evaluate_sampling_pair(
      model_sample_fn=lambda data: bfs_sampling.sample_bfs_categorical(data),
      true_sample_fn=lambda data: bfs_sampling.sample_bfs_categorical(data),
      model_input=[preds],
      true_input=outputs,
      adjacency=adjacency,
      source_nodes=source_nodes,
      validate_fn=bfs_validation.check_valid_bfs_tree,
  )
  random_sampling = evaluate_sampling_pair(
      model_sample_fn=lambda data: dfs_sampling.sample_random_list(data),
      true_sample_fn=lambda data: dfs_sampling.sample_random_list(data),
      model_input=[preds],
      true_input=outputs,
      adjacency=adjacency,
      source_nodes=source_nodes,
      validate_fn=bfs_validation.check_valid_bfs_tree,
  )
  prim = evaluate_sampling_pair(
      model_sample_fn=lambda data, s: bfs_sampling.sample_bfs_prim(data, s),
      true_sample_fn=lambda data, s: bfs_sampling.sample_bfs_prim(data, s),
      model_input=[preds],
      true_input=outputs,
      adjacency=adjacency,
      source_nodes=source_nodes,
      validate_fn=bfs_validation.check_valid_bfs_tree,
      s=source_nodes,
  )
  beam = evaluate_sampling_pair(
      model_sample_fn=lambda data, s: bfs_sampling.sample_bfs_beam(
          data, s, beam_width=3),
      true_sample_fn=lambda data, s: bfs_sampling.sample_bfs_beam(
          data, s, beam_width=3),
      model_input=[preds],
      true_input=outputs,
      adjacency=adjacency,
      source_nodes=source_nodes,
      validate_fn=bfs_validation.check_valid_bfs_tree,
      s=source_nodes,
  )

  result_dict = reports.build_bfs_result_dict(
      adjacency_flat=[i.flatten() for i in adjacency],
      source_nodes=source_nodes,
      categorical=categorical,
      random_sampling=random_sampling,
      prim=prim,
      beam=beam,
  )
  sampling_methods = {
      "Categorical": lambda data: bfs_sampling.sample_bfs_categorical(data),
      "Random": lambda data: dfs_sampling.sample_random_list(data),
      "Prim": lambda data: bfs_sampling.sample_bfs_prim(data, source_nodes),
      "Beam": lambda data: bfs_sampling.sample_bfs_beam(
          data, source_nodes, beam_width=3),
  }
  sampling_summary = distribution_validation.evaluate_sampling_methods(
      methods=sampling_methods,
      model_input=[preds],
      true_input=outputs,
      adjacency=adjacency,
      source_nodes=source_nodes,
      validate_fn=bfs_validation.check_valid_bfs_tree,
      n_samples=NSE,
  )
  result_dict.update(sampling_summary["result_dict"])
  if vd_flag:
    distribution_validation.save_sampling_curve_artifacts(
        sampling_summary["curves"],
        filename=f"{filename}_BFS",
        output_dir=output_dir,
    )
  report_sink = save_results_fn or reporting.discard_report
  report_sink(result_dict, f"{filename}_BFS")

  out.update(sampling_summary["scalar_metrics"])
  if extras:
    out.update(copy.deepcopy(extras))
  return {k: _unpack(v) for k, v in out.items()}


def _unpack(v):
  try:
    return v.item()
  except (AttributeError, ValueError):
    return v
