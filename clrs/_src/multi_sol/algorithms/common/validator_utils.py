"""Validator helpers for multi-solution extraction."""

from __future__ import annotations

import chex
import numpy as np


def is_square_adjacency(adjacency) -> bool:
  """Return whether adjacency is a rank-2 square matrix."""
  chex.assert_rank(adjacency, 2)
  return adjacency.shape[0] == adjacency.shape[1]


def coerce_parent_array(parent_tree, num_nodes):
  """Return int parent array, or None if shape/range is invalid."""
  if len(parent_tree) != num_nodes:
    return None
  parent_tree = np.asarray(parent_tree).astype(int)
  if np.any(parent_tree < 0) or np.any(parent_tree >= num_nodes):
    return None
  return parent_tree
