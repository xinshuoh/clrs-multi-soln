"""Definition objects for multi-solution algorithm extensions."""

from __future__ import annotations

import dataclasses
import numpy as np
from typing import Any, Callable, Sequence, Tuple

BatchExtractor = Callable[[Any], Tuple[Any, Any]]
ValidatorFn = Callable[[Any, object, int], bool]
ValidateFn = ValidatorFn
ExtractorFn = Callable[[Any, Any], Any]
ExtractFn = ExtractorFn
RunSingleFn = Callable[..., Any]


@dataclasses.dataclass(frozen=True)
class ExtractionMethod:
  """Stochastic extraction method for a solution distribution."""

  name: str
  model_distribution_sample: ExtractorFn
  target_distribution_sample: ExtractorFn

  @classmethod
  def same_sampler(
      cls,
      name: str,
      sample: ExtractorFn,
  ) -> "ExtractionMethod":
    return cls(
        name=name,
        model_distribution_sample=sample,
        target_distribution_sample=sample,
    )


@dataclasses.dataclass(frozen=True)
class AlgorithmBaseline:
  """Display metadata for symbolic algorithm-baseline evaluation."""

  name: str
  source_name: str = "Algorithm"


@dataclasses.dataclass(frozen=True)
class MultiSolTrainingConfig:
  """Configuration for empirical multi-solution training targets."""

  num_solutions: int = 20
  output_name: str = "pi"
  symbolic_sampler: "RandomizedAlgorithm | None" = None


@dataclasses.dataclass(frozen=True)
class RandomizedAlgorithm:
  """Randomized symbolic algorithm for one concrete solution."""

  sample_solution: RunSingleFn
  uses_source_node: bool = False

  def sample_one(self, adjacency, rng, source_node=None, deterministic=False):
    if self.uses_source_node:
      return self.sample_solution(adjacency, int(source_node), rng, deterministic=deterministic)
    del source_node
    return self.sample_solution(adjacency, rng, deterministic=deterministic)

  def sample_batch(self, adjacency, source_nodes, rng):
    if self.uses_source_node:
      return [
          self.sample_one(
              np.asarray(graph),
              rng,
              source_node=int(source),
          ) for graph, source in zip(adjacency, source_nodes)
      ]
    del source_nodes
    return [self.sample_one(np.asarray(graph), rng) for graph in adjacency]


@dataclasses.dataclass(frozen=True)
class MultiSolSolutionSpace:
  """Solution-space operations for one multi-solution algorithm."""

  batch_extractor: BatchExtractor
  validator: ValidateFn
  extraction_methods: Sequence[ExtractionMethod]
  algorithm_baseline: AlgorithmBaseline | None = None
  include_source_nodes: bool = False


@dataclasses.dataclass(frozen=True)
class MultiSolAlgorithm:
  """Metadata for one multi-solution algorithm."""

  name: str
  base_name: str
  solution_space: MultiSolSolutionSpace
  training: MultiSolTrainingConfig = dataclasses.field(default_factory=MultiSolTrainingConfig)
