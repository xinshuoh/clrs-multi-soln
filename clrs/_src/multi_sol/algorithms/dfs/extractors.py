"""DFS-specific stochastic extraction methods."""

from __future__ import annotations

import numpy as np

from clrs._src.multi_sol.algorithms.common import extractor_utils

def extract_argmax(outs_or_preds, _batch):
  trees = []
  for pred in outs_or_preds:
    for prob in pred["pi"].data:
      trees.append(np.argmax(prob, axis=1))
  return trees

def extract_argmax_true(outs_or_preds, _batch):
  trees = []
  for output in outs_or_preds:
    for prob in output.data:
      trees.append(np.argmax(prob, axis=1))
  return trees

def extract_random(outs_or_preds, _batch):
  trees = []
  for prob_matrix in extractor_utils.extract_prob_matrices(outs_or_preds):
    pi = [np.random.randint(len(row)) for row in prob_matrix]
    trees.append(pi)
  return trees

def extract_upwards(outs_or_preds, batch = None):
  return [single_sample_upwards(prob_matrix)
          for prob_matrix in extractor_utils.extract_prob_matrices(outs_or_preds)]
  
def single_sample_upwards(prob_matrix):
  """Sample one parent tree by repeatedly sampling upwards from leafy nodes."""
  prob_matrix = _row_wise_prob(np.asarray(prob_matrix))
  num_nodes = len(prob_matrix)
  pi = np.full(num_nodes, -1, dtype=int)

  for start in _leafiness_sort(prob_matrix):
    node = int(start)
    steps = 0
    while pi[node] == -1 and steps <= num_nodes:
      parent = _choose_uniformly(prob_matrix[node])
      if parent is None:
        parent = node
      parent = int(parent)
      pi[node] = parent
      node = parent
      steps += 1

    if steps > num_nodes and 0 <= node < num_nodes and pi[node] == -1:
      pi[node] = node

  missing = np.where(pi == -1)[0]
  pi[missing] = missing
  return pi
 
def _row_wise_prob(prob_matrix):
  return extractor_utils.normalize_rows(np.asarray(prob_matrix))

def _leafiness_sort(prob_matrix):
  """Most leafy node first (lowest parent-likelihood column sum)."""
  sums = np.sum(prob_matrix, axis=0)
  return np.argsort(sums)

def _choose_uniformly(not_prob_array):
  return extractor_utils.sample_index(not_prob_array, fallback="uniform")


def extract_alt_upwards(outs_or_preds, batch=None):
  return [get_parent_tree_upwards(prob_matrix)
          for prob_matrix in extractor_utils.extract_prob_matrices(outs_or_preds)]
  
def get_parent_tree_upwards(prob_matrix):
  """Explore upwards until all nodes have parents (possibly self-parent)."""
  parent_guesses = np.full(len(prob_matrix), np.inf)
  leafiness = _leafiness_sort(prob_matrix)
  for node_ix in leafiness:
    if parent_guesses[node_ix] == np.inf:
      parent_guesses = _explore_upwards(node_ix, parent_guesses, prob_matrix)
  if np.inf in parent_guesses:
    raise ValueError("not guessing parent for someone")
  return parent_guesses

def _explore_upwards(orphan_ix, parent_guesses, prob_matrix):
  while parent_guesses[orphan_ix] == np.inf:
    parent_guess = _choose_uniformly(prob_matrix[orphan_ix])
    parent_guesses[orphan_ix] = parent_guess
    orphan_ix = parent_guess
  return parent_guesses


EXTRACTORS = {
    "Argmax": extract_argmax,
    "Random": extract_random,
    "Upwards": extract_upwards,
    "altUpwards": extract_alt_upwards,
}

