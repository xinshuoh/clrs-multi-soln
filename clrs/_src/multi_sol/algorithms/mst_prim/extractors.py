"""MST-Prim-specific stochastic extraction methods."""

from __future__ import annotations

import numpy as np

from clrs._src.multi_sol.algorithms.dfs import extractors as dfs_extractors
from clrs._src.multi_sol.algorithms.common import extractor_utils
from clrs._src.multi_sol.interfaces import Extractor


def extract_argmax(outs_or_preds, _batch):
  return dfs_extractors.extract_argmax(outs_or_preds, _batch)

def extract_random(outs_or_preds, _batch):
  return dfs_extractors.extract_random(outs_or_preds, _batch)


def extract_tree(outs_or_preds, batch):
  """Sample source-rooted trees from parent probabilities over crossing edges."""
  trees = []
  for adjacency, source, prob_matrix in extractor_utils.iter_adjacency_source_prob_matrices(
      batch.adjacency, batch.source_nodes, outs_or_preds):
    trees.append(_mst_prim_tree_sampler(adjacency, source, prob_matrix))
  return trees

def _mst_prim_tree_sampler(adjacency, source, prob_matrix):
  """Grow a rooted spanning tree using P[v, u] on valid crossing edges."""
  adjacency = np.asarray(adjacency)
  prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
  num_nodes = prob_matrix.shape[0]
  pi = np.arange(num_nodes, dtype=int)
  pi[source] = source
  in_tree = np.zeros(num_nodes, dtype=bool)
  in_tree[source] = True

  while not np.all(in_tree):
    crossing_edges, edge_probs = _crossing_edges(adjacency, prob_matrix, in_tree)
    if not crossing_edges:
      break

    edge_ix: int = extractor_utils.sample_index(edge_probs, fallback="uniform")
    parent, child = crossing_edges[edge_ix]
    pi[child] = parent
    in_tree[child] = True

  return pi

def _crossing_edges(adjacency, prob_matrix, in_tree):
  crossing_edges = []
  edge_probs = []
  tree_nodes = np.where(in_tree)[0]
  remaining_nodes = np.where(~in_tree)[0]
  for parent in tree_nodes:
    for child in remaining_nodes:
      if adjacency[parent, child] != 0:
        crossing_edges.append((int(parent), int(child)))
        edge_probs.append(prob_matrix[child, parent])
  return crossing_edges, np.asarray(edge_probs, dtype=np.float64)


def extract_greedy(outs_or_preds, batch):
  """Sample MST-Prim parents with a Bellman-Ford-style greedy baseline."""
  trees = []
  for adjacency, source, prob_matrix in extractor_utils.iter_adjacency_source_prob_matrices(
      batch.adjacency, batch.source_nodes, outs_or_preds):
    trees.append(_mst_prim_greedy_sampler(
        adjacency,
        source,
        prob_matrix,
    ))
  return trees

def _mst_prim_greedy_sampler(
    adjacency,
    source,
    prob_matrix,
    num_candidates=3,
    max_resamples=10,
):
  """Choose low-weight sampled real neighbours independently for each node, with default number of candidates and resamples."""
  adjacency = np.asarray(adjacency)
  prob_matrix = extractor_utils.normalize_rows(np.asarray(prob_matrix))
  num_nodes = prob_matrix.shape[0]
  pi = np.arange(num_nodes, dtype=int)
  pi[source] = source

  for v in range(num_nodes):
    if v == source:
      continue

    chosen_parent = None
    for _ in range(max_resamples):
      candidates = extractor_utils.sample_indices(prob_matrix[v], num_candidates)
      plausible = [
          u for u in candidates
          if u != v and adjacency[u, v] != 0
      ]
      if plausible:
        weights = np.array([adjacency[u, v] for u in plausible])
        chosen_parent = int(plausible[int(np.argmin(weights))])
        break

    if chosen_parent is None:
      chosen_parent = extractor_utils.highest_probability_real_neighbour(
          adjacency, prob_matrix, v)
    pi[v] = chosen_parent

  return pi

EXTRACTORS = (
    Extractor("Argmax", extract_argmax, extract_argmax),
    Extractor("Random", extract_random, extract_random),
    Extractor("Tree", extract_tree, extract_tree),
    Extractor("Greedy", extract_greedy, extract_greedy),
)
