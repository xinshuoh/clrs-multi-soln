"""Dispatch helpers for optional multi-solution evaluation plugins."""

from __future__ import annotations

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
) -> Dict[str, Any]:
  """Evaluate using extension plugin when requested, else default evaluator."""
  if profile == "sampling" and extension_evaluator is not None:
    save_fn = (
        reporting.save_pickle_report if save_artifacts else reporting.discard_report
    )
    return extension_evaluator(
        sampler=sampler,
        predict_fn=predict_fn,
        sample_count=sample_count,
        rng_key=rng_key,
        extras=extras,
        save_results_fn=save_fn,
        filename=f"{artifact_prefix}_{algorithm_name}",
    )
  return fallback_eval_fn(
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      extras=extras,
  )


def _resolve_extension_evaluator(
    *, algorithm_name: str, profile: str
) -> Callable[..., Dict[str, Any]] | None:
  """Resolve extension evaluator only for extension-specific profiles."""
  if profile != "sampling":
    return None
  extension = _get_multisol_registry().get_extension(algorithm_name)
  return extension.evaluator if extension else None


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
  )
