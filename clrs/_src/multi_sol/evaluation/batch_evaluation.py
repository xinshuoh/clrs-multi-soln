"""Shared batch evaluation workflow for multi-solution algorithms."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np

from clrs._src.multi_sol.evaluation import distribution_validation
from clrs._src.multi_sol.evaluation import reporting
from clrs._src.multi_sol.evaluation.runners import evaluate_sampling_pair


BatchExtractor = Callable[[Any], Tuple[np.ndarray, np.ndarray]]
SampleFn = Callable[[Any, "EvaluationBatch"], Any]
ValidateFn = Callable[[np.ndarray, object, int], bool]
AlgorithmSampleFn = Callable[["EvaluationBatch", np.random.Generator], Any]


@dataclasses.dataclass(frozen=True)
class EvaluationBatch:
  """Collected model inputs and outputs for one multi-solution evaluation."""

  outputs: Any
  preds: Any
  adjacency: np.ndarray
  source_nodes: np.ndarray


@dataclasses.dataclass(frozen=True)
class SamplingMethod:
  """Model/target sampling pair used for validity and uniqueness evaluation."""

  name: str
  model_sample_fn: SampleFn
  true_sample_fn: SampleFn


@dataclasses.dataclass(frozen=True)
class AlgorithmSamplingSource:
  """Optional symbolic sampler used as a distribution-validation comparator."""

  method_name: str
  source_name: str
  sample_fn: AlgorithmSampleFn


def evaluate_multisol_batch(
    *,
    sampler,
    predict_fn,
    sample_count: int,
    rng_key,
    extras: Dict[str, Any],
    batch_extractor: BatchExtractor,
    validate_fn: ValidateFn,
    sampling_methods: Sequence[SamplingMethod],
    save_results_fn=None,
    filename: str,
    vd_flag: bool = False,
    n_samples: int = 100,
    output_dir: str = ".",
    curve_max_graphs: int | None = None,
    algorithm_source: AlgorithmSamplingSource | None = None,
    include_source_nodes: bool = False,
) -> Dict[str, Any]:
  """Collect predictions, evaluate samplers, and persist reports."""
  batch, out = _collect_batch(
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      batch_extractor=batch_extractor,
  )

  pair_results = _evaluate_sampling_pairs(
      batch=batch,
      sampling_methods=sampling_methods,
      validate_fn=validate_fn,
  )
  result_dict = _build_result_dict(
      batch=batch,
      pair_results=pair_results,
      include_source_nodes=include_source_nodes,
  )

  sampling_summary = _evaluate_sampling_distributions(
      batch=batch,
      sampling_methods=sampling_methods,
      validate_fn=validate_fn,
      n_samples=n_samples,
      curve_max_graphs=curve_max_graphs,
  )
  result_dict.update(sampling_summary["result_dict"])

  if vd_flag and algorithm_source is not None:
    algorithm_summary = _evaluate_algorithm_source(
        batch=batch,
        algorithm_source=algorithm_source,
        validate_fn=validate_fn,
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

  report_sink = save_results_fn or reporting.discard_report
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
    sampling_methods: Sequence[SamplingMethod],
    validate_fn: ValidateFn,
) -> Dict[str, Dict[str, object]]:
  results = {}
  for method in sampling_methods:
    results[method.name] = evaluate_sampling_pair(
        model_sample_fn=lambda data, method=method: method.model_sample_fn(
            data, batch),
        true_sample_fn=lambda data, method=method: method.true_sample_fn(
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
    sampling_methods: Sequence[SamplingMethod],
    validate_fn: ValidateFn,
    n_samples: int,
    curve_max_graphs: int | None,
) -> Dict[str, object]:
  model_methods = {}
  true_methods = {}
  for method in sampling_methods:
    model_methods[method.name] = (
        lambda data, method=method: method.model_sample_fn(data, batch))
    true_methods[method.name] = (
        lambda data, method=method: method.true_sample_fn(data, batch))
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
    algorithm_source: AlgorithmSamplingSource,
    validate_fn: ValidateFn,
    n_samples: int,
    curve_max_graphs: int | None,
) -> Dict[str, object]:
  algorithm_rng = np.random.default_rng(np.random.randint(0, 2**32))
  return distribution_validation.evaluate_sampling_sources(
      sources={
          (algorithm_source.method_name, algorithm_source.source_name): (
              lambda _data: algorithm_source.sample_fn(batch, algorithm_rng),
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
  from clrs._src.multi_sol.data import adapters
  return adapters.concat_tree

