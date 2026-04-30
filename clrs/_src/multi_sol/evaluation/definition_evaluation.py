"""Generic evaluation for multi-solution algorithm definitions."""

from __future__ import annotations

from typing import Any, Dict

from clrs._src.multi_sol.core import definitions
from clrs._src.multi_sol.evaluation import batch_evaluation


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
  if solution_space is None:
    raise ValueError(
        f"Algorithm definition '{definition.algorithm_name}' has no "
        "solution-space definition."
    )
  return batch_evaluation.evaluate_multisol_batch(
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      extras=extras,
      batch_extractor=solution_space.batch_extractor,
      validate_fn=solution_space.validation_method,
      sampling_methods=_to_sampling_methods(solution_space.extraction_methods),
      save_results_fn=save_results_fn,
      filename=filename or definition.algorithm_name,
      vd_flag=vd_flag,
      n_samples=NSE,
      output_dir=output_dir,
      curve_max_graphs=curve_max_graphs,
      algorithm_source=_to_algorithm_source(
          solution_space.generator_sampling_source),
      include_source_nodes=solution_space.include_source_nodes,
  )


def _to_sampling_methods(extraction_methods):
  return tuple(
      batch_evaluation.SamplingMethod(
          method.name,
          method.model_distribution_sample,
          method.target_distribution_sample,
      )
      for method in extraction_methods
  )


def _to_algorithm_source(generator_sampling_source):
  if generator_sampling_source is None:
    return None
  return batch_evaluation.AlgorithmSamplingSource(
      generator_sampling_source.name,
      generator_sampling_source.source_name,
      generator_sampling_source.sample_fn,
  )


def evaluator_for_definition(definition):
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
