"""Dispatch helpers for optional multi-solution evaluation plugins."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict

from clrs._src.multi_sol.evaluation import reporting


_MULTISOL_REGISTRY = None


def discard_results(_result_dict: Dict[str, Any], _filename: str) -> None:
  """Compatibility alias for side-effect-free report sink."""
  reporting.discard_report(_result_dict, _filename)


def save_results_artifact(result_dict: Dict[str, Any], filename: str) -> None:
  """Compatibility alias for pickle artifact sink."""
  reporting.save_pickle_report(result_dict, filename)


def _get_multisol_registry():
  global _MULTISOL_REGISTRY
  if _MULTISOL_REGISTRY is None:
    from clrs._src.multi_sol.core import registry as multisol_registry
    _MULTISOL_REGISTRY = multisol_registry
  return _MULTISOL_REGISTRY


def evaluate_with_optional_extension(
    *,
    algorithm_name: str,
    profile: str,
    extension_evaluator: Callable[..., Dict[str, Any]] | None,
    sampler,
    predict_fn,
    sample_count: int,
    rng_key,
    extras: Dict[str, Any],
    artifact_prefix: str,
    save_artifacts: bool,
    fallback_eval_fn: Callable[..., Dict[str, Any]],
    extension_kwargs: Dict[str, Any] | None = None,
    report_sink: Callable[[Dict[str, Any], str], None] | None = None,
) -> Dict[str, Any]:
  """Evaluate using extension plugin when requested, else default evaluator."""
  if profile == "sampling" and extension_evaluator is not None:
    save_fn = report_sink
    if save_fn is None:
      save_fn = (
          reporting.save_pickle_report
          if save_artifacts
          else reporting.discard_report
      )
    extension_call_kwargs = dict(
        sampler=sampler,
        predict_fn=predict_fn,
        sample_count=sample_count,
        rng_key=rng_key,
        extras=extras,
        save_results_fn=save_fn,
        filename=f"{artifact_prefix}_{algorithm_name}",
    )
    extension_call_kwargs.update(
        _filter_extension_kwargs(extension_evaluator, extension_kwargs)
    )
    return extension_evaluator(**extension_call_kwargs)
  return fallback_eval_fn(
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      extras=extras,
  )


def _filter_extension_kwargs(
    extension_evaluator: Callable[..., Dict[str, Any]],
    extension_kwargs: Dict[str, Any] | None,
) -> Dict[str, Any]:
  if not extension_kwargs:
    return {}
  try:
    signature = inspect.signature(extension_evaluator)
  except (TypeError, ValueError):
    return dict(extension_kwargs)

  if any(
      param.kind is inspect.Parameter.VAR_KEYWORD
      for param in signature.parameters.values()
  ):
    return dict(extension_kwargs)

  return {
      key: value
      for key, value in extension_kwargs.items()
      if key in signature.parameters
  }


def _resolve_extension_evaluator(
    *, algorithm_name: str, profile: str
) -> Callable[..., Dict[str, Any]] | None:
  """Resolve extension evaluator only for extension-specific profiles."""
  if profile != "sampling":
    return None
  extension = _get_multisol_registry().get_extension(algorithm_name)
  if extension is None:
    return None
  if getattr(extension, "evaluation", None) is not None:
    from clrs._src.multi_sol.evaluation import definition_evaluation
    return definition_evaluation.evaluator_for_definition(extension)
  return extension.evaluator


def evaluate_with_registry(
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
    extension_kwargs: Dict[str, Any] | None = None,
    report_sink: Callable[[Dict[str, Any], str], None] | None = None,
) -> Dict[str, Any]:
  """Evaluate by resolving optional extension evaluators from the registry."""
  extension_evaluator = _resolve_extension_evaluator(
      algorithm_name=algorithm_name, profile=profile
  )
  split_prefix = f"{artifact_prefix}_{split}"
  return evaluate_with_optional_extension(
      algorithm_name=algorithm_name,
      profile=profile,
      extension_evaluator=extension_evaluator,
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      extras=extras,
      artifact_prefix=split_prefix,
      save_artifacts=save_artifacts,
      fallback_eval_fn=fallback_eval_fn,
      extension_kwargs=extension_kwargs,
      report_sink=report_sink,
  )
