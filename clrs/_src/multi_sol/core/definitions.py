"""Definition objects for multi-solution algorithm extensions."""

from __future__ import annotations

import dataclasses
import numpy as np
from typing import Any, Callable, Dict, Optional, Sequence, Tuple


SpecFactory = Callable[[Dict[str, Dict[str, Any]]], Dict[str, Any]]
Algorithm = Callable[..., Any]
EvaluatorFn = Callable[..., dict]
SpecProvider = Dict[str, Any] | SpecFactory
BatchExtractor = Callable[[Any], Tuple[Any, Any]]
ValidateFn = Callable[[Any, object, int], bool]
ExtractFn = Callable[[Any, Any], Any]
GeneratorSampleFn = Callable[[Any, Any], Any]
RunSingleFn = Callable[..., Any]


@dataclasses.dataclass(frozen=True)
class ExtractionMethod:
  """Stochastic extraction method for a solution distribution."""

  name: str
  model_distribution_sample: ExtractFn
  target_distribution_sample: ExtractFn

  @classmethod
  def same_sampler(cls, name: str, sample: ExtractFn) -> "ExtractionMethod":
    return cls(
        name=name,
        model_distribution_sample=sample,
        target_distribution_sample=sample,
    )


@dataclasses.dataclass(frozen=True)
class GeneratorSamplingSource:
  """Symbolic generator source used as a solution-space comparator."""

  name: str
  source_name: str
  sample_fn: GeneratorSampleFn


@dataclasses.dataclass(frozen=True)
class TrainingDistribution:
  """Configuration for empirical multi-solution training targets."""

  num_solutions: int = 20
  output_name: str = "pi"


@dataclasses.dataclass(frozen=True)
class RandomizedAlgorithm:
  """Randomized symbolic algorithm for one concrete solution."""

  sample_solution: RunSingleFn
  uses_source_node: bool = False

  def sample_one(self, adjacency, rng, source_node=None, deterministic=False):
    if self.uses_source_node:
      return self.sample_solution(
          adjacency, int(source_node), rng, deterministic=deterministic)
    del source_node
    return self.sample_solution(adjacency, rng, deterministic=deterministic)

  def sample_batch(self, adjacency, source_nodes, rng):
    if self.uses_source_node:
      return [
          self.sample_one(
              np.asarray(graph),
              rng,
              source_node=int(source),
          )
          for graph, source in zip(adjacency, source_nodes)
      ]
    del source_nodes
    return [
        self.sample_one(np.asarray(graph), rng)
        for graph in adjacency
    ]


@dataclasses.dataclass(frozen=True)
class MultiSolSolutionSpace:
  """Solution-space operations for one multi-solution algorithm."""

  batch_extractor: BatchExtractor
  validation_method: ValidateFn
  extraction_methods: Sequence[ExtractionMethod]
  generator_sampling_source: GeneratorSamplingSource | None = None
  include_source_nodes: bool = False


@dataclasses.dataclass(frozen=True)
class MultiSolAlgorithm:
  """Single source of truth for one multi-solution algorithm."""

  algorithm_name: str
  base_algorithm_name: str
  spec: SpecProvider
  sampler_class: Optional[type] = None
  algorithm: Optional[Algorithm] = None
  evaluator: Optional[EvaluatorFn] = None
  training_distribution: TrainingDistribution = dataclasses.field(
      default_factory=TrainingDistribution)
  randomized_algorithm: Optional[RandomizedAlgorithm] = None
  solution_space: Optional[MultiSolSolutionSpace] = None
