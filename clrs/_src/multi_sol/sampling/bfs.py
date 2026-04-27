"""BFS extraction strategies (plugin-oriented)."""

from __future__ import annotations

import numpy as np

from clrs._src.multi_sol.sampling import base


def sample_bfs_prim(outs_or_preds, source_nodes):
  """Sample BFS trees using greedy processed-set attachment."""
  prob_matrix_list = base.extract_prob_matrices(outs_or_preds)
  source_nodes = base.as_index_list(source_nodes, len(prob_matrix_list))
  trees = []
  for i, prob_matrix in enumerate(prob_matrix_list):
    trees.append(prim_like_sampler(prob_matrix, int(source_nodes[i])))
  return trees


def prim_like_sampler(prob_matrix, source):
  num_nodes = prob_matrix.shape[0]
  pi = np.full(num_nodes, -1, dtype=int)
  pi[source] = source

  processed = {source}
  remaining = set(range(num_nodes)) - {source}

  while remaining:
    best_v = -1
    best_score = -1.0
    for v in remaining:
      score = sum(prob_matrix[v, u] for u in processed)
      if score > best_score:
        best_score = score
        best_v = v
    if best_v == -1:
      best_v = list(remaining)[0]

    candidate_parents = list(processed)
    probs = np.array([prob_matrix[best_v, u] for u in candidate_parents])
    parent_ix = base.sample_index(probs)
    if parent_ix is not None:
      parent = candidate_parents[parent_ix]
    else:
      parent = best_v

    pi[best_v] = parent
    processed.add(best_v)
    remaining.remove(best_v)

  return pi


def sample_bfs_categorical(outs_or_preds):
  """Sample each node parent independently using per-row categorical draw."""
  prob_matrix_list = base.extract_prob_matrices(outs_or_preds)
  trees = []
  for prob_matrix in prob_matrix_list:
    num_nodes = prob_matrix.shape[0]
    pi = np.zeros(num_nodes, dtype=int)
    normalized = base.normalize_rows(prob_matrix)
    for i in range(num_nodes):
      parent = base.sample_index(normalized[i])
      if parent is not None:
        pi[i] = parent
    trees.append(pi)
  return trees


def sample_bfs_beam(outs_or_preds, source_nodes, beam_width=3):
  """Sample BFS trees using heuristic beam search over processed-set states."""
  prob_matrix_list = base.extract_prob_matrices(outs_or_preds)
  source_nodes = base.as_index_list(source_nodes, len(prob_matrix_list))
  trees = []
  for i, prob_matrix in enumerate(prob_matrix_list):
    trees.append(bfs_beam_sampler(prob_matrix, int(source_nodes[i]), beam_width))
  return trees


def bfs_beam_sampler(prob_matrix, source, beam_width):
  num_nodes = prob_matrix.shape[0]
  initial_pi = np.full(num_nodes, -1, dtype=int)
  initial_pi[source] = source
  beam = [{"log_prob": 0.0, "pi": initial_pi, "processed": {source}}]

  for _ in range(num_nodes - 1):
    candidates = []
    for hyp in beam:
      processed = hyp["processed"]
      pi = hyp["pi"]
      curr_log_prob = hyp["log_prob"]
      remaining = set(range(num_nodes)) - processed
      if not remaining:
        candidates.append(hyp)
        continue

      best_v = -1
      best_mass = -1.0
      for v in remaining:
        mass = sum(prob_matrix[v, u] for u in processed)
        if mass > best_mass:
          best_mass = mass
          best_v = v
      if best_v == -1:
        best_v = list(remaining)[0]

      for u in list(processed):
        p_val = prob_matrix[best_v, u]
        if p_val > 1e-9:
          new_log_prob = curr_log_prob + np.log(p_val)
        else:
          new_log_prob = curr_log_prob - 1e9
        new_pi = pi.copy()
        new_pi[best_v] = u
        new_processed = processed.copy()
        new_processed.add(best_v)
        candidates.append(
            {"log_prob": new_log_prob, "pi": new_pi, "processed": new_processed}
        )

    if not candidates:
      break
    candidates.sort(key=lambda x: x["log_prob"], reverse=True)
    beam = candidates[:beam_width]

  return beam[0]["pi"]
