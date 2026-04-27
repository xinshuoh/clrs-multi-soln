"""Shared sampling helpers."""

from typing import Literal
from typing import List
from typing import overload

import numpy as np

from clrs._src.multi_sol.data.distribution import extract_probability_matrices


def normalize_rows(prob_matrix: np.ndarray) -> np.ndarray:
  """Row-normalize probability matrix while preserving zero rows."""
  out = prob_matrix.astype(np.float64, copy=True)
  row_sums = out.sum(axis=1)
  nonzero = row_sums > 0
  out[nonzero] = out[nonzero] / row_sums[nonzero][:, None]
  return out


def extract_prob_matrices(outs_or_preds) -> List[np.ndarray]:
  return extract_probability_matrices(outs_or_preds)


def as_instance_list(values, expected_len):
  values = np.asarray(values)
  if expected_len == 1 and values.ndim == 2:
    return [values]
  return values


def as_index_list(values, expected_len):
  values = np.asarray(values)
  if values.ndim == 0:
    return [int(values)] * expected_len
  return values


def iter_adjacency_source_prob_matrices(adjacencies, source_nodes, outs_or_preds):
  prob_matrix_list = extract_prob_matrices(outs_or_preds)
  adjacency_list = as_instance_list(adjacencies, len(prob_matrix_list))
  source_nodes = as_index_list(source_nodes, len(prob_matrix_list))
  for i, prob_matrix in enumerate(prob_matrix_list):
    yield adjacency_list[i], int(source_nodes[i]), prob_matrix


def normalized_probabilities(probabilities):
  probs = np.asarray(probabilities, dtype=np.float64)
  probs = np.where(np.isfinite(probs), probs, 0.0)
  probs = np.maximum(probs, 0.0)
  total = probs.sum()
  if total <= 0:
    return None
  return probs / total


@overload
def sample_index(probabilities, fallback: Literal["uniform"]) -> int:
  ...


@overload
def sample_index(probabilities, fallback: int) -> int:
  ...


@overload
def sample_index(probabilities, fallback: None = None) -> int | None:
  ...


def sample_index(probabilities, fallback=None):
  probs = normalized_probabilities(probabilities)
  if probs is None:
    if fallback == "uniform":
      return int(np.random.randint(len(probabilities)))
    return fallback
  return int(np.random.choice(len(probs), p=probs))


def sample_indices(probabilities, count):
  if count <= 0:
    return np.array([], dtype=int)
  probs = normalized_probabilities(probabilities)
  if probs is None:
    return np.array([], dtype=int)
  return np.random.choice(len(probs), size=count, p=probs)


def highest_probability_real_neighbour(adjacency, prob_matrix, node):
  neighbours = np.where(adjacency[:, node] != 0)[0]
  neighbours = neighbours[neighbours != node]
  if len(neighbours) == 0:
    return int(node)
  probs = np.asarray(prob_matrix)[node, neighbours]
  return int(neighbours[int(np.argmax(probs))])
