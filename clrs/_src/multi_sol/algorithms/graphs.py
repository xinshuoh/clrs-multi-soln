"""Randomized graph algorithms for multi-solution workflows."""

from __future__ import annotations

import numpy as np


def dfs_multi(A, rng, deterministic=False):
  """Sample one DFS parent tree using randomized tie-breaking."""
  del deterministic
  n = A.shape[0]
  color = np.zeros(n, dtype=np.int32)
  pi = np.arange(n, dtype=int)
  s_prev = np.arange(n, dtype=int)
  shuffled = rng.permutation(n)

  for s in range(n):
    if color[s] != 0:
      continue
    s_last = s
    u = s
    while True:
      if color[u] == 0:
        color[u] = 1

      for v in shuffled:
        if A[u, v] != 0 and color[v] == 0:
          pi[v] = u
          color[v] = 1
          s_prev[v] = s_last
          s_last = v
          break

      if s_last == u:
        color[u] = 2
        if s_prev[u] == u:
          break
        parent = s_prev[s_last]
        s_prev[s_last] = s_last
        s_last = parent

      u = s_last
  return pi


def bfs_multi(A, s, rng, deterministic=False):
  """Sample one BFS parent tree using randomized tie-breaking."""
  n = A.shape[0]
  reach = np.zeros(n, dtype=bool)
  pi = np.arange(n, dtype=int)
  reach[s] = True

  while True:
    prev_reach = np.copy(reach)
    sources = np.where(prev_reach)[0]
    if not deterministic:
      rng.shuffle(sources)
    for src in sources:
      for child in range(n):
        if A[src, child] > 0:
          if pi[child] == child and child != s:
            pi[child] = int(src)
          reach[child] = True
    if np.all(reach == prev_reach):
      break
  return pi


def bellman_ford_multi(A, s, rng, deterministic=False):
  """Sample one Bellman-Ford predecessor tree using randomized tie-breaking."""
  n = A.shape[0]
  d = np.zeros(n)
  pi = np.arange(n, dtype=int)
  msk = np.zeros(n)
  d[s] = 0
  msk[s] = 1

  if deterministic:
    shuffled_sources = np.concatenate(([0], np.arange(1, n)))
    shuffled_targets = np.arange(n)
  else:
    shuffled_sources = rng.permutation(n)
    shuffled_targets = rng.permutation(n)
  while True:
    prev_d = np.copy(d)
    prev_msk = np.copy(msk)
    for u in shuffled_sources:
      for v in shuffled_targets:
        if prev_msk[u] == 1 and A[u, v] != 0:
          if msk[v] == 0 or prev_d[u] + A[u, v] < d[v]:
            d[v] = prev_d[u] + A[u, v]
            pi[v] = int(u)
          msk[v] = 1
    if np.all(d == prev_d):
      break
  return pi


def mst_prim_multi(A, s, rng, deterministic=False):
  """Sample one Prim MST parent tree using randomized tie-breaking."""
  n = A.shape[0]
  key = np.zeros(n)
  mark = np.zeros(n)
  in_queue = np.zeros(n)
  pi = np.arange(n, dtype=int)
  key[s] = 0
  in_queue[s] = 1

  for _ in range(n):
    effective_keys = np.where(in_queue == 1, key, np.inf)
    min_key_val = np.min(effective_keys)
    if np.isinf(min_key_val):
      break
    candidates = np.where(effective_keys == min_key_val)[0]
    u = int(candidates[0] if deterministic else rng.choice(candidates))
    if in_queue[u] == 0:
      break
    mark[u] = 1
    in_queue[u] = 0
    for v in range(n):
      if A[u, v] != 0:
        if mark[v] == 0 and (in_queue[v] == 0 or A[u, v] < key[v]):
          pi[v] = u
          key[v] = A[u, v]
          in_queue[v] = 1
  return pi
