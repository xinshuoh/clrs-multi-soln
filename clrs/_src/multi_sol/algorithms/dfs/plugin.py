"""DFS plugin wiring for modular multi-solution evaluation."""

from typing import Dict

import clrs
import jax

from clrs._src.multi_sol.data.adapters import concat_tree
from clrs._src.multi_sol.evaluation import distribution_validation
from clrs._src.multi_sol.evaluation import reporting
from clrs._src.multi_sol.evaluation import reports
from clrs._src.multi_sol.evaluation.runners import evaluate_sampling_pair
from clrs._src.multi_sol.sampling import dfs as dfs_sampling
from clrs._src.multi_sol.validation import dfs as dfs_validation
from clrs._src import dfs_uniqueness_check


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
  if vd_flag:
    distribution_validation.run_dfs_distribution_validation(
        adjacency=adjacency,
        outputs=outputs,
        pred_batches=pred_batches,
        nse=NSE,
    )
  source_nodes = [0] * len(adjacency)
  out = clrs.evaluate(outputs, preds)

  random_sampling = evaluate_sampling_pair(
      model_sample_fn=lambda data: dfs_sampling.sample_random_list(data),
      true_sample_fn=lambda data: dfs_sampling.sample_random_list(data),
      model_input=[preds],
      true_input=outputs,
      adjacency=adjacency,
      source_nodes=source_nodes,
      validate_fn=dfs_validation.check_valid_dfs_tree,
  )
  argmax_sampling = evaluate_sampling_pair(
      model_sample_fn=lambda data: dfs_sampling.sample_argmax_listofdict(data),
      true_sample_fn=lambda data: dfs_sampling.sample_argmax_listofdatapoint(data),
      model_input=[preds],
      true_input=outputs,
      adjacency=adjacency,
      source_nodes=source_nodes,
      validate_fn=dfs_validation.check_valid_dfs_tree,
  )
  upwards_sampling = evaluate_sampling_pair(
      model_sample_fn=lambda data: dfs_sampling.sample_upwards(data),
      true_sample_fn=lambda data: dfs_sampling.sample_upwards(data),
      model_input=[preds],
      true_input=outputs,
      adjacency=adjacency,
      source_nodes=source_nodes,
      validate_fn=dfs_validation.check_valid_dfs_tree,
  )
  alt_upwards_sampling = evaluate_sampling_pair(
      model_sample_fn=lambda data: dfs_sampling.sample_altUpwards(data),
      true_sample_fn=lambda data: dfs_sampling.sample_altUpwards(data),
      model_input=[preds],
      true_input=outputs,
      adjacency=adjacency,
      source_nodes=source_nodes,
      validate_fn=dfs_validation.check_valid_dfs_tree,
  )

  result_dict = reports.build_dfs_result_dict(
      adjacency_flat=[i.flatten() for i in adjacency],
      argmax=argmax_sampling,
      random_sampling=random_sampling,
      upwards=upwards_sampling,
      alt_upwards=alt_upwards_sampling,
  )
  model_up_uniques, model_up_valids_uniques, model_up_valids = (
      dfs_uniqueness_check.check_uniqueness_dfs(adjacency, [preds]))
  true_up_uniques, true_up_valids_uniques, true_up_valids = (
      dfs_uniqueness_check.check_uniqueness_dfs(adjacency, outputs))
  model_alt_uniques, model_alt_valids_uniques, model_alt_valids = (
      dfs_uniqueness_check.check_uniqueness_dfs(
          adjacency, [preds], method="altupwards"))
  true_alt_uniques, true_alt_valids_uniques, true_alt_valids = (
      dfs_uniqueness_check.check_uniqueness_dfs(
          adjacency, outputs, method="altupwards"))
  result_dict.update({
      "Upwards_Model_Uniques": model_up_uniques,
      "Upwards_Model_Valids_Uniques": model_up_valids_uniques,
      "Upwards_Model_Valids": model_up_valids,
      "Upwards_True_Uniques": true_up_uniques,
      "Upwards_True_Valids_Uniques": true_up_valids_uniques,
      "Upwards_True_Valids": true_up_valids,
      "altUpwards_Model_Uniques": model_alt_uniques,
      "altUpwards_Model_Valids_Uniques": model_alt_valids_uniques,
      "altUpwards_Model_Valids": model_alt_valids,
      "altUpwards_True_Uniques": true_alt_uniques,
      "altUpwards_True_Valids_Uniques": true_alt_valids_uniques,
      "altUpwards_True_Valids": true_alt_valids,
  })
  report_sink = save_results_fn or reporting.discard_report
  report_sink(result_dict, f"{filename}_DFS")

  if extras:
    out.update(extras)
  return {k: _unpack(v) for k, v in out.items()}


def _unpack(v):
  try:
    return v.item()
  except (AttributeError, ValueError):
    return v
