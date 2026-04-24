"""DFS sampling methods for multi-solution extraction."""

from __future__ import annotations

import numpy as np

from clrs._src.multi_sol.data.distribution import extract_probability_matrices


def sample_argmax(outs_or_preds):
  trees = []
  for prob_matrix in extract_probability_matrices(outs_or_preds):
    trees.append(np.argmax(prob_matrix, axis=1))
  return trees


def sample_argmax_listofdict(preds):
  trees = []
  for pred in preds:
    for prob in pred["pi"].data:
      trees.append(np.argmax(prob, axis=1))
  return trees


def sample_argmax_listofdatapoint(outputs):
  trees = []
  for output in outputs:
    for prob in output.data:
      trees.append(np.argmax(prob, axis=1))
  return trees


def sample_random_list(outs_or_preds):
  trees = []
  rng = np.random.default_rng()
  for prob_matrix in extract_probability_matrices(outs_or_preds):
    pi = [rng.integers(len(row)) for row in prob_matrix]
    trees.append(pi)
  return trees


def leafiness_sort(prob_matrix):
  """Most leafy node first (lowest parent-likelihood column sum)."""
  sums = np.sum(prob_matrix, axis=0)
  return np.argsort(sums)


def row_wise_prob(prob_matrix):
  normalized = np.array(prob_matrix, dtype=np.float64, copy=True)
  for row_ix in range(len(normalized)):
    row_sum = normalized[row_ix].sum()
    if row_sum != 0:
      normalized[row_ix] = normalized[row_ix] / row_sum
  return normalized


def choose_uniformly(not_prob_array):
  val = np.random.uniform(low=0, high=sum(not_prob_array))
  sums = np.cumsum(not_prob_array)
  for threshold_ix in range(len(sums)):
    if val < sums[threshold_ix]:
      return threshold_ix
  return np.random.randint(len(not_prob_array))


# Legacy compatibility aliases.
leafinessSort = leafiness_sort
rowWiseProb = row_wise_prob
chooseUniformly = choose_uniformly


def single_sample_upwards(prob_matrix):
  """Sample one parent tree by repeatedly sampling upwards from leafy nodes."""
  prob_matrix = np.array(prob_matrix, copy=True)
  leafiness = np.asarray(leafiness_sort(prob_matrix))
  pi = np.full(len(prob_matrix), np.inf)
  while sum(leafiness) > -len(leafiness):
    altered_prob_matrix = row_wise_prob(prob_matrix)
    leaf = leafiness[leafiness != -1][0]
    parent = choose_uniformly(altered_prob_matrix[leaf])
    pi[leaf] = parent
    leafiness[leaf] = -1
    leafiness[parent] = -1
    altered_prob_matrix[:, leaf] = 0

    while pi[parent] == np.inf:
      leaf = parent
      parent = choose_uniformly(altered_prob_matrix[leaf])
      pi[leaf] = parent
      leafiness[leaf] = -1
      leafiness[parent] = -1
      altered_prob_matrix[:, leaf] = 0

  if sum(np.isin(pi, np.inf)) > 0:
    raise ValueError("Leaf with no parent")
  return pi


def sample_upwards(outs_or_preds):
  return [single_sample_upwards(prob_matrix)
          for prob_matrix in extract_probability_matrices(outs_or_preds)]


def explore_upwards(orphan_ix, parent_guesses, prob_matrix):
  while parent_guesses[orphan_ix] == np.inf:
    parent_guess = choose_uniformly(prob_matrix[orphan_ix])
    parent_guesses[orphan_ix] = parent_guess
    orphan_ix = parent_guess
  return parent_guesses


def get_parent_tree_upwards(prob_matrix):
  """Explore upwards until all nodes have parents (possibly self-parent)."""
  parent_guesses = np.full(len(prob_matrix), np.inf)
  leafiness = leafiness_sort(prob_matrix)
  for node_ix in leafiness:
    if parent_guesses[node_ix] == np.inf:
      parent_guesses = explore_upwards(node_ix, parent_guesses, prob_matrix)
  if np.inf in parent_guesses:
    raise ValueError("not guessing parent for someone")
  return parent_guesses


def sample_altUpwards(outs_or_preds):
  return [get_parent_tree_upwards(prob_matrix)
          for prob_matrix in extract_probability_matrices(outs_or_preds)]


def extract_probMatrices(outs_or_preds):
  """Compatibility alias for legacy call sites."""
  return extract_probability_matrices(outs_or_preds)

__all__ = (
    "sample_argmax",
    "sample_argmax_listofdict",
    "sample_argmax_listofdatapoint",
    "sample_random_list",
    "sample_upwards",
    "sample_altUpwards",
    "single_sample_upwards",
    "get_parent_tree_upwards",
    "extract_probMatrices",
    "choose_uniformly",
    "leafinessSort",
    "rowWiseProb",
    "chooseUniformly",
)
