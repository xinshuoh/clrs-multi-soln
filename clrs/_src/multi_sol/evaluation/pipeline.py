"""Unified evaluation workflow for multi-solution algorithms."""

from __future__ import annotations

import copy
import dataclasses
import inspect
from typing import Any, Callable, Dict, Sequence, Tuple

from absl import logging
import numpy as np

from clrs._src.multi_sol.core import definitions
from clrs._src.multi_sol.evaluation import artifacts
from clrs._src.multi_sol.evaluation import sampling_metrics


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


_MULTISOL_REGISTRY = None


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


def evaluate_definition(
    *,
    definition: definitions.MultiSolAlgorithm,
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
  solution_space = definition.solution_space
  return evaluate_sampling_batch(
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      extras=extras,
      batch_extractor=solution_space.batch_extractor,
      validate_fn=solution_space.validator,
      sampling_methods=_to_sampling_methods(solution_space.extraction_methods),
      save_results_fn=save_results_fn,
      filename=filename or definition.name,
      vd_flag=vd_flag,
      n_samples=NSE,
      output_dir=output_dir,
      curve_max_graphs=curve_max_graphs,
      algorithm_source=_to_algorithm_source(definition),
      include_source_nodes=solution_space.include_source_nodes,
  )


def build_definition_evaluator(definition):
  """Return a callable evaluator bound to a definition object."""

  def _evaluate(
      *,
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
  ):
    return evaluate_definition(
        definition=definition,
        sampler=sampler,
        predict_fn=predict_fn,
        sample_count=sample_count,
        rng_key=rng_key,
        extras=extras,
        save_results_fn=save_results_fn,
        filename=filename,
        vd_flag=vd_flag,
        NSE=NSE,
        output_dir=output_dir,
        curve_max_graphs=curve_max_graphs,
    )

  return _evaluate


def evaluate_with_optional_sampling(
    *,
    algorithm_name: str,
    profile: str,
    sampling_evaluator: Callable[..., Dict[str, Any]] | None,
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
  """Evaluate using sampling evaluator when requested, else fallback."""
  if profile == "sampling" and sampling_evaluator is not None:
    save_fn = report_sink
    if save_fn is None:
      save_fn = (
          artifacts.save_pickle_report
          if save_artifacts
          else artifacts.discard_report
      )
    sampling_call_kwargs = dict(
        sampler=sampler,
        predict_fn=predict_fn,
        sample_count=sample_count,
        rng_key=rng_key,
        extras=extras,
        save_results_fn=save_fn,
        filename=f"{artifact_prefix}_{algorithm_name}",
    )
    sampling_call_kwargs.update(
        _filter_sampling_kwargs(sampling_evaluator, sampling_kwargs)
    )
    return sampling_evaluator(**sampling_call_kwargs)
  return fallback_eval_fn(
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      extras=extras,
  )


def evaluate_with_sampling_registry(
    *,
    algorithm_name: str,
    split: str,
    profile: str,
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
  """Evaluate by resolving optional sampling evaluators from the registry."""
  sampling_evaluator = _resolve_sampling_evaluator(
      algorithm_name=algorithm_name, profile=profile
  )
  split_prefix = f"{artifact_prefix}_{split}"
  return evaluate_with_optional_sampling(
      algorithm_name=algorithm_name,
      profile=profile,
      sampling_evaluator=sampling_evaluator,
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      extras=extras,
      artifact_prefix=split_prefix,
      save_artifacts=save_artifacts,
      fallback_eval_fn=fallback_eval_fn,
      sampling_kwargs=sampling_kwargs,
      report_sink=report_sink,
  )


def evaluate_sampling_batch(
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
  logging.info(
      'Multi-solution evaluation: collecting %d examples.', sample_count)
  batch, out = _collect_batch(
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      batch_extractor=batch_extractor,
  )

  logging.info('Multi-solution evaluation: evaluating one-shot samplers.')
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

  logging.info(
      'Multi-solution evaluation: evaluating sampling distributions with '
      '%d samples per method.',
      n_samples,
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
    logging.info(
        'Multi-solution evaluation: evaluating algorithm-source sampler.')
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
    sampling_metrics.save_sampling_curve_artifacts(
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
    sampling_methods: Sequence[SamplingMethod],
    validate_fn: ValidateFn,
) -> Dict[str, Dict[str, object]]:
  results = {}
  for method in sampling_methods:
    logging.info(
        'Multi-solution evaluation: evaluating one-shot method %s.',
        method.name,
    )
    results[method.name] = _evaluate_sampling_pair(
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
  return sampling_metrics.evaluate_mixed_sampling_methods(
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
  return sampling_metrics.evaluate_sampling_sources(
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
  from clrs._src.multi_sol.algorithms.common import batch_extractors
  return batch_extractors.concat_tree


def _to_sampling_methods(extraction_methods):
  return tuple(
      SamplingMethod(
          method.name,
          method.model_distribution_sample,
          method.target_distribution_sample,
      )
      for method in extraction_methods
  )


def _to_algorithm_source(definition):
  algorithm_baseline = definition.solution_space.algorithm_baseline
  symbolic_sampler = definition.training.symbolic_sampler
  if algorithm_baseline is None or symbolic_sampler is None:
    return None
  def _sample(batch, rng):
    source_nodes = batch.source_nodes if symbolic_sampler.uses_source_node else None
    return symbolic_sampler.sample_batch(batch.adjacency, source_nodes, rng)
  return AlgorithmSamplingSource(
      algorithm_baseline.name,
      algorithm_baseline.source_name,
      _sample,
  )


def _get_multisol_registry():
  global _MULTISOL_REGISTRY
  if _MULTISOL_REGISTRY is None:
    from clrs._src.multi_sol import registry as multisol_registry
    _MULTISOL_REGISTRY = multisol_registry
  return _MULTISOL_REGISTRY


def _resolve_sampling_evaluator(
    *,
    algorithm_name: str,
    profile: str,
) -> Callable[..., Dict[str, Any]] | None:
  if profile != "sampling":
    return None
  extension = _get_multisol_registry().get(algorithm_name)
  if extension is None:
    return None
  return build_definition_evaluator(extension)


def _filter_sampling_kwargs(
    sampling_evaluator: Callable[..., Dict[str, Any]],
    sampling_kwargs: Dict[str, Any] | None,
) -> Dict[str, Any]:
  if not sampling_kwargs:
    return {}
  try:
    signature = inspect.signature(sampling_evaluator)
  except (TypeError, ValueError):
    return dict(sampling_kwargs)

  if any(
      param.kind is inspect.Parameter.VAR_KEYWORD
      for param in signature.parameters.values()
  ):
    return dict(sampling_kwargs)

  return {
      key: value
      for key, value in sampling_kwargs.items()
      if key in signature.parameters
  }
