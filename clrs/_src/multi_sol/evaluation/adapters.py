"""Adapters to extract algorithm-specific arrays from CLRS feedback."""

from __future__ import annotations

from typing import Tuple

import jax
import numpy as np


def concat_tree(dps, axis):
  """Tree-concatenate CLRS structured arrays."""
  return jax.tree_util.tree_map(lambda *x: np.concatenate(x, axis), *dps)


def extract_bfs_graph_and_source(feedback) -> Tuple[np.ndarray, np.ndarray]:
  """Extract BFS adjacency matrix and source index arrays from CLRS feedback."""
  return _extract_graph_with_source(feedback)


def extract_bellman_ford_graph_and_source(
    feedback,
) -> Tuple[np.ndarray, np.ndarray]:
  """Extract Bellman-Ford adjacency matrix and source index arrays."""
  return _extract_graph_with_source(feedback)


def extract_mst_prim_graph_and_source(feedback) -> Tuple[np.ndarray, np.ndarray]:
  """Extract MST-Prim adjacency matrix and source index arrays."""
  return _extract_graph_with_source(feedback)


def extract_dfs_graph_and_source(feedback) -> Tuple[np.ndarray, np.ndarray]:
  """Extract DFS adjacency matrix and synthetic source index arrays."""
  adjacency = feedback[0][0][1].data
  source = np.zeros(adjacency.shape[0], dtype=int)
  return adjacency, source


def _extract_graph_with_source(feedback) -> Tuple[np.ndarray, np.ndarray]:
  adjacency = feedback[0][0][2].data
  source = np.argmax(feedback[0][0][1].data, axis=1)
  return adjacency, source
