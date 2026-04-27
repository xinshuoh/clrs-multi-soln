"""MST-Prim plugin wiring for modular multi-solution evaluation."""

from typing import Dict

import clrs
import jax
import numpy as np

from clrs._src.multi_sol.data.adapters import concat_tree
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
  del vd_flag, NSE, output_dir
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
  model_tree_uniques, model_tree_valids_uniques, model_tree_valids = (
      _sampling_uniqueness(
          [preds], adjacency, source_nodes, mst_sampling.sample_mst_prim_tree))
  true_tree_uniques, true_tree_valids_uniques, true_tree_valids = (
      _sampling_uniqueness(
          outputs, adjacency, source_nodes, mst_sampling.sample_mst_prim_tree))
  model_greedy_uniques, model_greedy_valids_uniques, model_greedy_valids = (
      _sampling_uniqueness(
          [preds], adjacency, source_nodes, mst_sampling.sample_mst_prim_greedy))
  true_greedy_uniques, true_greedy_valids_uniques, true_greedy_valids = (
      _sampling_uniqueness(
          outputs, adjacency, source_nodes, mst_sampling.sample_mst_prim_greedy))
  result_dict.update({
      "Tree_Model_Uniques": model_tree_uniques,
      "Tree_Model_Valids_Uniques": model_tree_valids_uniques,
      "Tree_Model_Valids": model_tree_valids,
      "Tree_True_Uniques": true_tree_uniques,
      "Tree_True_Valids_Uniques": true_tree_valids_uniques,
      "Tree_True_Valids": true_tree_valids,
      "Greedy_Model_Uniques": model_greedy_uniques,
      "Greedy_Model_Valids_Uniques": model_greedy_valids_uniques,
      "Greedy_Model_Valids": model_greedy_valids,
      "Greedy_True_Uniques": true_greedy_uniques,
      "Greedy_True_Valids_Uniques": true_greedy_valids_uniques,
      "Greedy_True_Valids": true_greedy_valids,
  })
  report_sink = save_results_fn or reporting.discard_report
  report_sink(result_dict, f"{filename}_MSTPrim")

  if extras:
    out.update(extras)
  return {k: _unpack(v) for k, v in out.items()}


def _sampling_uniqueness(
    outs_or_preds,
    adjacency,
    source_nodes,
    sample_fn,
    n_samples=5,
):
  samples = np.asarray([
      sample_fn(adjacency, source_nodes, outs_or_preds)
      for _ in range(n_samples)
  ])
  uniques = []
  valids_uniques = []
  valids = []
  for i in range(len(adjacency)):
    samples_for_graph = samples[:, i]
    unique_trees = [
        list(item) for item in set(tuple(row) for row in samples_for_graph)
    ]
    uniques.append(len(unique_trees) / n_samples)
    unique_valids = [
        mst_validation.check_valid_mst_prim_tree(
            adjacency[i], tree, int(source_nodes[i]))
        for tree in unique_trees
    ]
    valids_uniques.append(sum(unique_valids) / len(unique_trees))
    sample_valids = [
        mst_validation.check_valid_mst_prim_tree(
            adjacency[i], tree, int(source_nodes[i]))
        for tree in samples_for_graph
    ]
    valids.append(sum(sample_valids) / n_samples)
  return uniques, valids_uniques, valids


def _unpack(v):
  try:
    return v.item()
  except (AttributeError, ValueError):
    return v
