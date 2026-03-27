"""Adapters to extract algorithm-specific arrays from CLRS feedback."""

from typing import Tuple

import jax
import numpy as np


def concat_tree(dps, axis):
  """Tree-concatenate CLRS structured arrays."""
  return jax.tree_util.tree_map(lambda *x: np.concatenate(x, axis), *dps)


def extract_bfs_graph_and_source(feedback) -> Tuple[np.ndarray, np.ndarray]:
  """Extract BFS adjacency matrix and source index arrays from CLRS feedback."""
  adjacency = feedback[0][0][2].data
  source = np.argmax(feedback[0][0][1].data, axis=1)
  return adjacency, source

