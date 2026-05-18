"""Interface objects for multi-solution algorithms."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence, Tuple

import numpy as np


@dataclasses.dataclass(frozen=True)
class EvaluationBatch:
  """Collected model inputs and outputs for one multi-solution evaluation."""

  outputs: Any
  preds: Any
  adjacency: np.ndarray
  source_nodes: np.ndarray


BatchExtractor = Callable[[Any], Tuple[np.ndarray, np.ndarray]]
ValidatorFn = Callable[[np.ndarray, object, int], bool]
ExtractorFn = Callable[[Any, EvaluationBatch], Any]
RunSingleFn = Callable[..., Any]


@dataclasses.dataclass(frozen=True)
class Extractor:
  """Stochastic extraction method for a solution distribution."""

  name: str
  model_sample: ExtractorFn
  target_sample: ExtractorFn


@dataclasses.dataclass(frozen=True)
class ReferenceSampler:
  """Display metadata for reference-sampler evaluation."""

  name: str
  source_name: str = "Algorithm"


@dataclasses.dataclass(frozen=True)
class Generator:
  """Generator config and symbolic sampler for one algorithm."""

  sample_target: RunSingleFn
  sample_solution: RunSingleFn
  num_solutions: int = 20
  output_name: str = "pi"
  uses_source_node: bool = False

  def target(self, *args, **kwargs):
    kwargs.setdefault("num_solutions", self.num_solutions)
    kwargs.setdefault("output_name", self.output_name)
    return self.sample_target(*args, **kwargs)

  def sample_one(self, adjacency, rng, source_node=None, deterministic=False):
    if self.uses_source_node:
      return self.sample_solution(
          adjacency,
          int(source_node),
          rng,
          deterministic=deterministic,
      )
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
class MultiSolAlgorithm:
  """Definition object for a multi-solution algorithm."""

  name: str
  base_name: str
  generator: Generator
  batch_extractor: BatchExtractor
  validator: ValidatorFn
  extractors: Sequence[Extractor]
  reference_sampler: ReferenceSampler | None = None
  include_source_nodes_in_report: bool = False
