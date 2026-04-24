"""Shared helpers for multi-solution graph target generators."""

from __future__ import annotations

import numpy as np

from clrs._src import specs
from clrs._src.multi_sol.core import registry as multisol_registry


def resolve_multisol_spec(algorithm_name: str) -> specs.Spec:
  return multisol_registry.resolve_specs(specs.SPECS)[algorithm_name]


def parent_distribution_from_trees(parent_trees, num_nodes: int) -> np.ndarray:
  """Convert sampled parent trees into an empirical parent distribution."""
  parent_mats = []
  for tree in parent_trees:
    mat = np.zeros((num_nodes, num_nodes))
    for node in range(num_nodes):
      mat[node, tree[node]] = 1
    parent_mats.append(mat)
  return sum(parent_mats) / len(parent_mats)

