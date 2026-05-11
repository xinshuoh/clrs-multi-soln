"""Unified evaluation workflow for multi-solution algorithms."""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Sequence, Tuple

from absl import logging
import numpy as np

from clrs._src.multi_sol.evaluation import artifacts
from clrs._src.multi_sol.evaluation import distribution_validation
from clrs._src.multi_sol.interfaces import BatchExtractor
from clrs._src.multi_sol.interfaces import EvaluationBatch
from clrs._src.multi_sol.interfaces import Extractor
from clrs._src.multi_sol.interfaces import MultiSolAlgorithm
from clrs._src.multi_sol.interfaces import ValidatorFn


def _accuracy(mask) -> float:
  return float(sum(mask)) / float(len(mask)) if mask else 0.0


def _evaluate_sampling_pair(
    *,
    model_sample_fn: Callable,
    true_sample_fn: Callable,
    model_input,
    true_input,
    adjacency,
    source_nodes,
    validate_fn: Callable,
    **kwargs,
) -> Dict[str, object]:
  model_trees = model_sample_fn(model_input, **kwargs)
  true_trees = true_sample_fn(true_input, **kwargs)
  model_mask = [
      validate_fn(adjacency[i], model_trees[i], int(source_nodes[i]))
      for i in range(len(model_trees))
  ]
  true_mask = [
      validate_fn(adjacency[i], true_trees[i], int(source_nodes[i]))
      for i in range(len(true_trees))
  ]
  return {
      "model_trees": model_trees,
      "true_trees": true_trees,
      "model_mask": model_mask,
      "true_mask": true_mask,
      "model_accuracy": _accuracy(model_mask),
      "true_accuracy": _accuracy(true_mask),
  }


def evaluate_algorithm(
    *,
    algorithm: MultiSolAlgorithm,
    sampler,
    predict_fn,
    sample_count,
    rng_key,
    extras,
    save_results_fn=None,
    filename=None,
    vd_flag=False,
    NSE=100,
    output_dir=".",
    curve_max_graphs=None,
) -> Dict[str, Any]:
  """Evaluate a multi-solution algorithm from its definition object."""
  return _evaluate_sampling_batch(
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      extras=extras,
      algorithm=algorithm,
      save_results_fn=save_results_fn,
      filename=filename or algorithm.name,
      vd_flag=vd_flag,
      n_samples=NSE,
      output_dir=output_dir,
      curve_max_graphs=curve_max_graphs,
  )


def evaluate_with_optional_sampling(
    *,
    algorithm_name: str,
    profile: str,
    multi_sol_algorithm: MultiSolAlgorithm | None,
    sampler,
    predict_fn,
    sample_count: int,
    rng_key,
    extras: Dict[str, Any],
    artifact_prefix: str,
    save_artifacts: bool,
    fallback_eval_fn: Callable[..., Dict[str, Any]],
    sampling_kwargs: Dict[str, Any] | None = None,
    report_sink: Callable[[Dict[str, Any], str], None] | None = None,
) -> Dict[str, Any]:
  """Evaluate using multi-solution sampling when requested, else fallback."""
  if profile == "sampling" and multi_sol_algorithm is not None:
    save_fn = report_sink
    if save_fn is None:
      save_fn = (
          artifacts.save_pickle_report
          if save_artifacts
          else artifacts.discard_report
      )
    sampling_call_kwargs = dict(
        algorithm=multi_sol_algorithm,
        sampler=sampler,
        predict_fn=predict_fn,
        sample_count=sample_count,
        rng_key=rng_key,
        extras=extras,
        save_results_fn=save_fn,
        filename=f"{artifact_prefix}_{algorithm_name}",
    )
    if sampling_kwargs:
      sampling_call_kwargs.update(dict(sampling_kwargs))
    return evaluate_algorithm(**sampling_call_kwargs)
  return fallback_eval_fn(
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      extras=extras,
  )


def _evaluate_sampling_batch(
    *,
    sampler,
    predict_fn,
    sample_count: int,
    rng_key,
    extras: Dict[str, Any],
    algorithm: MultiSolAlgorithm,
    save_results_fn=None,
    filename: str,
    vd_flag: bool = False,
    n_samples: int = 100,
    output_dir: str = ".",
    curve_max_graphs: int | None = None,
) -> Dict[str, Any]:
  """Collect predictions, evaluate samplers, and persist reports."""
  logging.info(
      'Multi-solution evaluation: collecting %d examples.', sample_count)
  batch, out = _collect_batch(
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      batch_extractor=algorithm.batch_extractor,
  )

  logging.info('Multi-solution evaluation: evaluating one-shot samplers.')
  pair_results = _evaluate_sampling_pairs(
      batch=batch,
      extractors=algorithm.extractors,
      validate_fn=algorithm.validator,
  )
  result_dict = _build_result_dict(
      batch=batch,
      pair_results=pair_results,
      include_source_nodes=algorithm.include_source_nodes_in_report,
  )

  logging.info(
      'Multi-solution evaluation: evaluating sampling distributions with '
      '%d samples per method.',
      n_samples,
  )
  sampling_summary = _evaluate_sampling_distributions(
      batch=batch,
      extractors=algorithm.extractors,
      validate_fn=algorithm.validator,
      n_samples=n_samples,
      curve_max_graphs=curve_max_graphs,
  )
  result_dict.update(sampling_summary["result_dict"])

  if vd_flag and algorithm.reference_sampler is not None:
    logging.info(
        'Multi-solution evaluation: evaluating algorithm-source sampler.')
    algorithm_summary = _evaluate_algorithm_source(
        batch=batch,
        algorithm=algorithm,
        validate_fn=algorithm.validator,
        n_samples=n_samples,
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

  report_sink = save_results_fn or artifacts.discard_report
  logging.info('Multi-solution evaluation: writing sampling report %s.', filename)
  report_sink(result_dict, filename)

  out.update(sampling_summary["scalar_metrics"])
  if extras:
    out.update(copy.deepcopy(extras))
  return {key: _unpack(value) for key, value in out.items()}


def _collect_batch(
    *,
    sampler,
    predict_fn,
    sample_count: int,
    rng_key,
    batch_extractor: BatchExtractor,
) -> Tuple[EvaluationBatch, Dict[str, Any]]:
  processed_samples = 0
  pred_batches = []
  outputs = []
  adjacency_batches = []
  source_batches = []

  while processed_samples < sample_count:
    feedback = next(sampler)
    batch_size = feedback.outputs[0].data.shape[0]
    outputs.append(feedback.outputs)
    jax_module = _jax()
    new_rng_key, rng_key = jax_module.random.split(rng_key)
    cur_preds, _ = predict_fn(new_rng_key, feedback.features)
    pred_batches.append(cur_preds)
    processed_samples += batch_size

    adjacency, source_nodes = batch_extractor(feedback)
    adjacency_batches.append(adjacency)
    source_batches.append(source_nodes)
    logging.info(
        'Multi-solution evaluation: collected %d/%d examples.',
        min(processed_samples, sample_count),
        sample_count,
    )

  concat_tree = _concat_tree()
  outputs = concat_tree(outputs, axis=0)
  adjacency = concat_tree(adjacency_batches, axis=0)
  source_nodes = concat_tree(source_batches, axis=0).astype(int)
  preds = concat_tree(pred_batches, axis=0)
  return (
      EvaluationBatch(
          outputs=outputs,
          preds=preds,
          adjacency=adjacency,
          source_nodes=source_nodes,
      ),
      _clrs().evaluate(outputs, preds),
  )


def _evaluate_sampling_pairs(
    *,
    batch: EvaluationBatch,
    extractors: Sequence[Extractor],
    validate_fn: ValidatorFn,
) -> Dict[str, Dict[str, object]]:
  results = {}
  for extractor in extractors:
    logging.info(
        'Multi-solution evaluation: evaluating one-shot method %s.',
        extractor.name,
    )
    results[extractor.name] = _evaluate_sampling_pair(
        model_sample_fn=lambda data, extractor=extractor: extractor.model_sample(
            data, batch),
        true_sample_fn=lambda data, extractor=extractor: extractor.target_sample(
            data, batch),
        model_input=[batch.preds],
        true_input=batch.outputs,
        adjacency=batch.adjacency,
        source_nodes=batch.source_nodes,
        validate_fn=validate_fn,
    )
  return results


def _evaluate_sampling_distributions(
    *,
    batch: EvaluationBatch,
    extractors: Sequence[Extractor],
    validate_fn: ValidatorFn,
    n_samples: int,
    curve_max_graphs: int | None,
) -> Dict[str, object]:
  model_methods = {}
  true_methods = {}
  for extractor in extractors:
    model_methods[extractor.name] = (
        lambda data, extractor=extractor: extractor.model_sample(data, batch))
    true_methods[extractor.name] = (
        lambda data, extractor=extractor: extractor.target_sample(data, batch))
  return distribution_validation.evaluate_mixed_sampling_methods(
      model_methods=model_methods,
      true_methods=true_methods,
      model_input=[batch.preds],
      true_input=batch.outputs,
      adjacency=batch.adjacency,
      source_nodes=batch.source_nodes,
      validate_fn=validate_fn,
      n_samples=n_samples,
      curve_max_graphs=curve_max_graphs,
  )


def _evaluate_algorithm_source(
    *,
    batch: EvaluationBatch,
    algorithm: MultiSolAlgorithm,
    validate_fn: ValidatorFn,
    n_samples: int,
    curve_max_graphs: int | None,
) -> Dict[str, object]:
  reference_sampler = algorithm.reference_sampler
  if reference_sampler is None:
    return {"result_dict": {}, "scalar_metrics": {}, "curves": []}
  generator = algorithm.generator
  algorithm_rng = np.random.default_rng(np.random.randint(0, 2**32))
  source_nodes = batch.source_nodes if generator.uses_source_node else None
  return distribution_validation.evaluate_sampling_sources(
      sources={
          (reference_sampler.name, reference_sampler.source_name): (
              lambda _data: generator.sample_batch(
                  batch.adjacency, source_nodes, algorithm_rng),
              None,
          ),
      },
      adjacency=batch.adjacency,
      source_nodes=batch.source_nodes,
      validate_fn=validate_fn,
      n_samples=n_samples,
      curve_max_graphs=curve_max_graphs,
  )


def _build_result_dict(
    *,
    batch: EvaluationBatch,
    pair_results: Dict[str, Dict[str, object]],
    include_source_nodes: bool,
) -> Dict[str, object]:
  result_dict: Dict[str, object] = {
      "As": [adjacency.flatten() for adjacency in batch.adjacency],
  }
  if include_source_nodes:
    result_dict["Source_Nodes"] = batch.source_nodes.tolist()

  for method_name, result in pair_results.items():
    result_dict.update(_sampling_pair_report_fields(method_name, result))
  return result_dict


def _sampling_pair_report_fields(
    method_name: str,
    result: Dict[str, object],
) -> Dict[str, object]:
  return {
      f"{method_name}_Model_Trees": result["model_trees"],
      f"{method_name}_True_Trees": result["true_trees"],
      f"{method_name}_Model_Mask": result["model_mask"],
      f"{method_name}_True_Mask": result["true_mask"],
      f"{method_name}_Model_Accuracy": result["model_accuracy"],
      f"{method_name}_True_Accuracy": result["true_accuracy"],
  }


def _unpack(value):
  try:
    return value.item()
  except (AttributeError, ValueError):
    return value


def _jax():
  import jax  # Imported lazily so tests can install lightweight stubs.
  return jax


def _clrs():
  import clrs  # Imported lazily so plugin tests can install lightweight stubs.
  return clrs


def _concat_tree():
  from clrs._src.multi_sol.algorithms.common import batch_extractors
  return batch_extractors.concat_tree
