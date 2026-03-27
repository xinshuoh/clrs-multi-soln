"""BFS plugin wiring for modular multi-solution evaluation."""

import copy
from typing import Dict

import clrs
import jax
import numpy as np

from clrs._src import dfs_sampling
from clrs._src.multi_sol.data.adapters import concat_tree
from clrs._src.multi_sol.evaluation import reports
from clrs._src.multi_sol.evaluation.runners import evaluate_sampling_pair
from clrs._src.multi_sol.sampling import bfs as bfs_sampling
from clrs._src.multi_sol.validation import bfs as bfs_validation


def evaluate_bfs_multisol_batch(
    sampler,
    predict_fn,
    sample_count,
    rng_key,
    extras,
    save_results_fn,
    filename="bfs_accuracy",
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
    adjacency_batches.append(feedback[0][0][2].data)
    source_batches.append(np.argmax(feedback[0][0][1].data, axis=1))

  outputs = concat_tree(outputs, axis=0)
  adjacency = concat_tree(adjacency_batches, axis=0)
  source_nodes = concat_tree(source_batches, axis=0)
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
  save_results_fn(result_dict, f"{filename}_BFS")

  if extras:
    out.update(copy.deepcopy(extras))
  return {k: _unpack(v) for k, v in out.items()}


def _unpack(v):
  try:
    return v.item()
  except (AttributeError, ValueError):
    return v

