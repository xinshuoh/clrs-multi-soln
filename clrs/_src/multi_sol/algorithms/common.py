"""Shared helpers for multi-solution graph target generators."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from clrs._src import probing
from clrs._src import specs
from clrs._src.multi_sol.core import definitions
from clrs._src.multi_sol.core import registry as multisol_registry


SingleExecution = Callable[
    [np.random.RandomState, specs.Spec, bool],
    Tuple[np.ndarray, probing.ProbesDict],
]


def resolve_multisol_spec(algorithm_name: str) -> specs.Spec:
  return multisol_registry.resolve_specs(specs.SPECS)[algorithm_name]


def generate_parent_distribution_target(
    *,
    algorithm_name: str | None = None,
    algorithm_spec: specs.Spec | None = None,
    training_distribution: definitions.TrainingDistribution | None = None,
    num_nodes: int,
    seed: int,
    deterministic: bool,
    run_single: SingleExecution,
    num_solutions: int | None = None,
) -> Tuple[np.ndarray, probing.ProbesDict]:
  """Run repeated symbolic executions and expose a parent distribution target."""
  if training_distribution is None:
    training_distribution = _resolve_training_distribution(
        algorithm_name, num_solutions)
  if algorithm_spec is None:
    if algorithm_name is None:
      raise ValueError("Provide either algorithm_spec or algorithm_name.")
    algorithm_spec = resolve_multisol_spec(algorithm_name)
  rng = np.random.RandomState(seed)
  parent_trees = []
  probes_list = []

  repetitions = 1 if deterministic else training_distribution.num_solutions
  for _ in range(repetitions):
    parent_tree, probes = run_single(rng, algorithm_spec, deterministic)
    parent_trees.append(parent_tree)
    probes_list.append(probes)

  parent_dist = parent_distribution_from_trees(parent_trees, num_nodes)
  probes_list[0]['output']['node'][training_distribution.output_name][
      'data'] = parent_dist
  return parent_dist, probes_list[0]


def _resolve_training_distribution(
    algorithm_name: str | None,
    num_solutions: int | None,
) -> definitions.TrainingDistribution:
  if num_solutions is not None:
    return definitions.TrainingDistribution(num_solutions=num_solutions)
  if algorithm_name is not None:
    extension = multisol_registry.get_extension(algorithm_name)
    if extension is not None:
      return extension.training_distribution
  return definitions.TrainingDistribution()




def parent_distribution_from_trees(parent_trees, num_nodes: int) -> np.ndarray:
  """Convert sampled parent trees into an empirical parent distribution."""
  parent_mats = []
  for tree in parent_trees:
    mat = np.zeros((num_nodes, num_nodes))
    for node in range(num_nodes):
      mat[node, tree[node]] = 1
    parent_mats.append(mat)
  return sum(parent_mats) / len(parent_mats)
