"""BFS-specific stochastic extraction methods."""

from __future__ import annotations

import numpy as np

from clrs._src.multi_sol.algorithms.common import extractor_utils
from clrs._src.multi_sol.interfaces import Extractor


def extract_categorical(outs_or_preds, batch):
  """Sample each node parent independently using per-row categorical draw."""
  prob_matrix_list = extractor_utils.extract_prob_matrices(outs_or_preds)
  source_nodes = extractor_utils.as_index_list(batch.source_nodes, len(prob_matrix_list))
  trees = []
  for instance, prob_matrix in enumerate(prob_matrix_list):
    num_nodes = prob_matrix.shape[0]
    pi = np.arange(num_nodes)
    source = int(source_nodes[instance])
    pi[source] = source
    normalized = extractor_utils.normalize_rows(prob_matrix)
    for i in range(num_nodes):
      if i == source:
        continue
      parent = extractor_utils.sample_index(normalized[i])
      if parent is not None:
        pi[i] = parent
    pi[source] = source
    trees.append(pi)
  return trees


def extract_random(outs_or_preds, _batch):
  trees = []
  for prob_matrix in extractor_utils.extract_prob_matrices(outs_or_preds):
    pi = [np.random.randint(len(row)) for row in prob_matrix]
    trees.append(pi)
  return trees


def extract_prim(outs_or_preds, batch):
  """Sample BFS trees using greedy processed-set attachment."""
  prob_matrix_list = extractor_utils.extract_prob_matrices(outs_or_preds)
  source_nodes = extractor_utils.as_index_list(batch.source_nodes, len(prob_matrix_list))
  trees = []
  for i, prob_matrix in enumerate(prob_matrix_list):
    trees.append(_prim_like_sampler(prob_matrix, int(source_nodes[i])))
  return trees


def _prim_like_sampler(prob_matrix, source):
  num_nodes = prob_matrix.shape[0]
  pi = np.arange(num_nodes)
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
    processed_probs = np.array([prob_matrix[best_v, u] for u in candidate_parents])
    if not np.any(processed_probs > 0.0):
      candidate_parents = [best_v]
    probs = np.array([prob_matrix[best_v, u] for u in candidate_parents])
    parent_ix = extractor_utils.sample_index(probs)
    if parent_ix is not None:
      parent = candidate_parents[parent_ix]
    else:
      parent = best_v

    pi[best_v] = parent
    processed.add(best_v)
    remaining.remove(best_v)

  return pi


def extract_wave(outs_or_preds, batch):
  """Sample BFS trees by expanding predicted reachability waves."""
  prob_matrix_list = extractor_utils.extract_prob_matrices(outs_or_preds)
  source_nodes = extractor_utils.as_index_list(batch.source_nodes, len(prob_matrix_list))
  trees = []
  for i, prob_matrix in enumerate(prob_matrix_list):
    trees.append(_wave_sampler(prob_matrix, int(source_nodes[i])))
  return trees


def _wave_sampler(prob_matrix, source):
  num_nodes = prob_matrix.shape[0]
  pi = np.arange(num_nodes)
  pi[source] = source

  reached = {source}
  unreached = set(range(num_nodes)) - {source}

  while unreached:
    reached_list = sorted(reached)
    next_wave = []
    for node in sorted(unreached):
      parent_probs = np.asarray(
          [prob_matrix[node, parent] for parent in reached_list],
          dtype=np.float64,
      )
      parent_probs = np.where(np.isfinite(parent_probs), parent_probs, 0.0)
      parent_probs = np.maximum(parent_probs, 0.0)
      self_parent_prob = prob_matrix[node, node]
      if not np.isfinite(self_parent_prob):
        self_parent_prob = 0.0
      self_parent_prob = max(float(self_parent_prob), 0.0)
      if np.sum(parent_probs) > self_parent_prob:
        parent_ix = extractor_utils.sample_index(parent_probs)
        if parent_ix is not None:
          pi[node] = reached_list[parent_ix]
          next_wave.append(node)

    if not next_wave:
      break

    reached.update(next_wave)
    unreached.difference_update(next_wave)

  pi[source] = source
  return pi


def extract_beam(outs_or_preds, batch, beam_width: int = 3):
  """Sample BFS trees using stochastic beam search over processed-set states."""
  prob_matrix_list = extractor_utils.extract_prob_matrices(outs_or_preds)
  source_nodes = extractor_utils.as_index_list(batch.source_nodes, len(prob_matrix_list))
  trees = []
  for i, prob_matrix in enumerate(prob_matrix_list):
    trees.append(_bfs_beam_sampler(prob_matrix, int(source_nodes[i]), beam_width))
  return trees


def _sample_parent_candidate(prob_matrix, node, candidate_parents):
  """Sample one parent candidate from predicted mass over `candidate_parents`."""
  candidate_parents = list(candidate_parents)
  if not candidate_parents:
    return None

  parent_probs = np.asarray(
      [prob_matrix[node, parent] for parent in candidate_parents],
      dtype=np.float64,
  )
  parent_probs = np.where(np.isfinite(parent_probs), parent_probs, 0.0)
  parent_probs = np.maximum(parent_probs, 0.0)
  parent_ix = extractor_utils.sample_index(parent_probs, fallback="uniform")
  return candidate_parents[parent_ix]


def _bfs_beam_sampler(prob_matrix, source, beam_width):
  num_nodes = prob_matrix.shape[0]
  initial_pi = np.arange(num_nodes)
  initial_pi[source] = source
  beam_width = max(1, beam_width)
  beam = [
      {"log_prob": 0.0, "pi": initial_pi.copy(), "processed": {source}}
      for _ in range(beam_width)
  ]

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

      candidate_parents = list(processed)
      processed_probs = np.array([prob_matrix[best_v, u] for u in candidate_parents])
      if not np.any(processed_probs > 0.0):
        candidate_parents = [best_v]
      u = _sample_parent_candidate(prob_matrix, best_v, candidate_parents)
      if u is None:
        continue
      p_val = prob_matrix[best_v, u]
      if p_val > 0.0:
        new_log_prob = curr_log_prob + np.log(p_val)
      else:
        new_log_prob = -np.inf
      new_pi = pi.copy()
      new_pi[source] = source
      new_pi[best_v] = u
      new_processed = processed.copy()
      new_processed.add(best_v)
      candidates.append({"log_prob": new_log_prob, "pi": new_pi, "processed": new_processed})

    if not candidates:
      break
    candidates.sort(key=lambda x: x["log_prob"], reverse=True)
    beam = candidates[:beam_width]

  best_pi = beam[0]["pi"].copy()
  best_pi[source] = source
  return best_pi


EXTRACTORS = (
    Extractor("Categorical", extract_categorical, extract_categorical),
    Extractor("Random", extract_random, extract_random),
    Extractor("Prim", extract_prim, extract_prim),
    Extractor("Wave", extract_wave, extract_wave),
    # Extractor("Beam", extract_beam, extract_beam),
)
