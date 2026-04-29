"""Shared helpers for multi-solution graph target generators."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from clrs._src import probing
from clrs._src import specs
from clrs._src.multi_sol.core import registry as multisol_registry


SingleExecution = Callable[
    [np.random.RandomState, specs.Spec, bool],
    Tuple[np.ndarray, probing.ProbesDict],
]


def resolve_multisol_spec(algorithm_name: str) -> specs.Spec:
  return multisol_registry.resolve_specs(specs.SPECS)[algorithm_name]


def generate_parent_distribution_target(
    *,
    algorithm_name: str,
    num_nodes: int,
    seed: int,
    deterministic: bool,
    run_single: SingleExecution,
    num_solutions: int = 20,
) -> Tuple[np.ndarray, probing.ProbesDict]:
  """Run repeated symbolic executions and expose a parent distribution target."""
  rng = np.random.RandomState(seed)
  algorithm_spec = resolve_multisol_spec(algorithm_name)
  parent_trees = []
  probes_list = []

  repetitions = 1 if deterministic else num_solutions
  for _ in range(repetitions):
    parent_tree, probes = run_single(rng, algorithm_spec, deterministic)
    parent_trees.append(parent_tree)
    probes_list.append(probes)

  parent_dist = parent_distribution_from_trees(parent_trees, num_nodes)
  probes_list[0]['output']['node']['pi']['data'] = parent_dist
  return parent_dist, probes_list[0]


def parent_distribution_from_trees(parent_trees, num_nodes: int) -> np.ndarray:
  """Convert sampled parent trees into an empirical parent distribution."""
  parent_mats = []
  for tree in parent_trees:
    mat = np.zeros((num_nodes, num_nodes))
    for node in range(num_nodes):
      mat[node, tree[node]] = 1
    parent_mats.append(mat)
  return sum(parent_mats) / len(parent_mats)
