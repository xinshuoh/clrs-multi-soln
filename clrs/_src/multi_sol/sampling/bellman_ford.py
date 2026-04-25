"""Bellman-Ford sampling methods for multi-solution extraction."""

from __future__ import annotations

import numpy as np

from clrs._src.multi_sol.sampling import dfs as dfs_sampling
from clrs._src.multi_sol.data.distribution import extract_probability_matrices


def sample_beamsearch(adjacencies, source_nodes, outs_or_preds):
  prob_matrix_list = extract_probability_matrices(outs_or_preds)
  pi_trees = []
  for ix, prob_matrix in enumerate(prob_matrix_list):
    adjacency = adjacencies[ix]
    source = source_nodes[ix]
    pi_trees.append(BF_beamsearch(adjacency, source, prob_matrix))
  return pi_trees


def BF_beamsearch(adjacency, source, prob_matrix, beamwidth=3):
  """Beam search sampler for Bellman-Ford parent distributions."""
  pi = np.arange(len(prob_matrix))
  pi[source] = source

  for target in range(len(prob_matrix)):
    if target == source:
      continue
    candidates_rev = [[target] for _ in range(beamwidth)]
    candidates_cost = [0 for _ in range(beamwidth)]
    best_path_cost = np.inf
    best_path_stemming_from_source = None

    for _ in range(len(prob_matrix)):
      longer_paths = []
      longer_path_costs = []

      for candidate_ix, candidate_path in enumerate(candidates_rev):
        highest_node = candidate_path[-1]
        parent_probs = prob_matrix[highest_node]
        for _ in range(beamwidth):
          candidate_parent = dfs_sampling.choose_uniformly(parent_probs)
          new_path = np.append(candidate_path, candidate_parent)
          longer_paths.append(new_path)

          cost_of_new_edge = adjacency[candidate_parent, highest_node]
          if cost_of_new_edge == 0:
            cost_of_new_edge = np.inf
          prev_cost = candidates_cost[candidate_ix]
          longer_path_costs.append(prev_cost + cost_of_new_edge)

      for path_ix, path in enumerate(longer_paths):
        if path[-1] == source and longer_path_costs[path_ix] < best_path_cost:
          best_path_stemming_from_source = path
          best_path_cost = longer_path_costs[path_ix]

      path_ixs_by_lowest_cost = np.argsort(longer_path_costs)
      best_path_ixs = path_ixs_by_lowest_cost[:beamwidth]
      candidates_rev = np.array(longer_paths, dtype=object)[best_path_ixs]
      candidates_cost = np.array(longer_path_costs)[best_path_ixs].tolist()

    if best_path_stemming_from_source is not None:
      pi[target] = best_path_stemming_from_source[1]

  return pi


def sample_greedysearch(adjacencies, source_nodes, outs_or_preds):
  prob_matrix_list = extract_probability_matrices(outs_or_preds)
  pi_trees = []
  for ix, prob_matrix in enumerate(prob_matrix_list):
    adjacency = adjacencies[ix]
    source = source_nodes[ix]
    pi_trees.append(BF_greedysearch(adjacency, source, prob_matrix))
  return pi_trees


def BF_greedysearch(adjacency, source, prob_matrix, beamwidth=3):
  """Greedy parent sampler constrained by edge existence in adjacency."""
  pi = np.zeros(len(prob_matrix))
  pi[source] = source

  for node in range(len(prob_matrix)):
    if node == source:
      continue
    candidates_costs = np.full(beamwidth, np.inf)
    tries = 0
    while (candidates_costs == np.full(len(candidates_costs), np.inf)).all() and tries < 10:
      candidates = [
          dfs_sampling.choose_uniformly(prob_matrix[node]) for _ in range(beamwidth)
      ]
      candidates_costs = [adjacency[candidate, node] for candidate in candidates]
      for ix in range(len(candidates_costs)):
        if candidates_costs[ix] == 0:
          candidates_costs[ix] = np.inf
      tries += 1
    pi[node] = candidates[np.argmin(candidates_costs)]
  return pi


__all__ = (
    "sample_beamsearch",
    "sample_greedysearch",
    "BF_beamsearch",
    "BF_greedysearch",
)
