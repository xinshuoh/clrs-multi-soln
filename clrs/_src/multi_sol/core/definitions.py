"""Definition objects for multi-solution algorithm extensions."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional, Sequence, Tuple


SpecFactory = Callable[[Dict[str, Dict[str, Any]]], Dict[str, Any]]
Algorithm = Callable[..., Any]
EvaluatorFn = Callable[..., dict]
SpecProvider = Dict[str, Any] | SpecFactory
BatchExtractor = Callable[[Any], Tuple[Any, Any]]
ValidateFn = Callable[[Any, object, int], bool]


@dataclasses.dataclass(frozen=True)
class MultiSolSolutionSpace:
  """Solution-space operations for one multi-solution algorithm."""

  batch_extractor: BatchExtractor
  validation_method: ValidateFn
  extraction_methods: Sequence[Any]
  generator_sampling_source: Any = None
  include_source_nodes: bool = False

  @property
  def validate_fn(self):
    """Compatibility alias for older evaluation wiring."""
    return self.validation_method

  @property
  def sampling_methods(self):
    """Compatibility alias for older evaluation wiring."""
    return self.extraction_methods

  @property
  def algorithm_source(self):
    """Compatibility alias for older evaluation wiring."""
    return self.generator_sampling_source


@dataclasses.dataclass(frozen=True)
class MultiSolAlgorithmDefinition:
  """Single source of truth for one multi-solution algorithm."""

  algorithm_name: str
  base_algorithm_name: str
  spec: SpecProvider
  sampler_class: Optional[type] = None
  algorithm: Optional[Algorithm] = None
  evaluator: Optional[EvaluatorFn] = None
  solution_space: Optional[MultiSolSolutionSpace] = None

  @property
  def evaluation(self):
    """Compatibility alias for older definition-driven evaluator wiring."""
    return self.solution_space


# Compatibility names used by the existing registry and tests.
MultiSolExtensionDefinition = MultiSolAlgorithmDefinition
MultiSolAlgorithmExtension = MultiSolAlgorithmDefinition
MultiSolEvaluationDefinition = MultiSolSolutionSpace
