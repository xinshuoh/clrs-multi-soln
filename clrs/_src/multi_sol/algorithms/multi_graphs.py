"""Multi-solution graph-target generators.

This module keeps multi-solution target generation out of the base CLRS graph
algorithm implementations. Base wrappers in `clrs._src.algorithms.graphs`
delegate here for compatibility.
"""

from typing import Tuple

import chex
from clrs._src import probing
from clrs._src import specs
import numpy as np

_Array = np.ndarray
_Out = Tuple[_Array, probing.ProbesDict]
_NUM_SOLUTIONS = 20


def _parent_distribution_from_trees(parent_trees, num_nodes: int) -> _Array:
  """Converts sampled parent trees into an empirical parent distribution."""
  parent_mats = []
  for tree in parent_trees:
    mat = np.zeros((num_nodes, num_nodes))
    for node in range(num_nodes):
      mat[node, tree[node]] = 1
    parent_mats.append(mat)
  return sum(parent_mats) / len(parent_mats)


def dfs_multi(A: _Array, seed: int, deterministic: bool = False) -> _Out:
  """Multiple-solution depth-first search target generation."""
  rng = np.random.RandomState(seed)
  chex.assert_rank(A, 2)
  probeslist = []
  pies = []

  num_solutions = 1 if deterministic else _NUM_SOLUTIONS
  for _ in range(num_solutions):
    probes = probing.initialize(specs.SPECS['dfs_multi'])

    A_pos = np.arange(A.shape[0])
    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            'A': np.copy(A),
            'adj': probing.graph(np.copy(A))
        })

    color = np.zeros(A.shape[0], dtype=np.int32)
    pi = np.arange(A.shape[0])
    d = np.zeros(A.shape[0])
    f = np.zeros(A.shape[0])
    s_prev = np.arange(A.shape[0])
    time = 0

    shuffled = np.arange(A.shape[0])
    rng.shuffle(shuffled)

    for s in range(A.shape[0]):
      if color[s] == 0:
        s_last = s
        u = s
        v = s
        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'pi_h': np.copy(pi),
                'color': probing.array_cat(color, 3),
                'd': np.copy(d),
                'f': np.copy(f),
                's_prev': np.copy(s_prev),
                's': probing.mask_one(s, A.shape[0]),
                'u': probing.mask_one(u, A.shape[0]),
                'v': probing.mask_one(v, A.shape[0]),
                's_last': probing.mask_one(s_last, A.shape[0]),
                'time': time
            })
        while True:
          if color[u] == 0 or d[u] == 0.0:
            time += 0.01
            d[u] = time
            color[u] = 1
            probing.push(
                probes,
                specs.Stage.HINT,
                next_probe={
                    'pi_h': np.copy(pi),
                    'color': probing.array_cat(color, 3),
                    'd': np.copy(d),
                    'f': np.copy(f),
                    's_prev': np.copy(s_prev),
                    's': probing.mask_one(s, A.shape[0]),
                    'u': probing.mask_one(u, A.shape[0]),
                    'v': probing.mask_one(v, A.shape[0]),
                    's_last': probing.mask_one(s_last, A.shape[0]),
                    'time': time
                })

          for v in shuffled:
            if A[u, v] != 0 and color[v] == 0:
              pi[v] = u
              color[v] = 1
              s_prev[v] = s_last
              s_last = v
              probing.push(
                  probes,
                  specs.Stage.HINT,
                  next_probe={
                      'pi_h': np.copy(pi),
                      'color': probing.array_cat(color, 3),
                      'd': np.copy(d),
                      'f': np.copy(f),
                      's_prev': np.copy(s_prev),
                      's': probing.mask_one(s, A.shape[0]),
                      'u': probing.mask_one(u, A.shape[0]),
                      'v': probing.mask_one(v, A.shape[0]),
                      's_last': probing.mask_one(s_last, A.shape[0]),
                      'time': time
                  })
              break

          if s_last == u:
            color[u] = 2
            time += 0.01
            f[u] = time
            probing.push(
                probes,
                specs.Stage.HINT,
                next_probe={
                    'pi_h': np.copy(pi),
                    'color': probing.array_cat(color, 3),
                    'd': np.copy(d),
                    'f': np.copy(f),
                    's_prev': np.copy(s_prev),
                    's': probing.mask_one(s, A.shape[0]),
                    'u': probing.mask_one(u, A.shape[0]),
                    'v': probing.mask_one(v, A.shape[0]),
                    's_last': probing.mask_one(s_last, A.shape[0]),
                    'time': time
                })
            if s_prev[u] == u:
              assert s_prev[s_last] == s_last
              break
            pr = s_prev[s_last]
            s_prev[s_last] = s_last
            s_last = pr

          u = s_last

    probing.push(probes, specs.Stage.OUTPUT, next_probe={'pi': np.copy(pi)})
    probing.finalize(probes)
    pies.append(pi)
    probeslist.append(probes)

  parent_dist = _parent_distribution_from_trees(pies, A.shape[0])
  probeslist[0]['output']['node']['pi']['data'] = parent_dist
  return parent_dist, probeslist[0]


def bfs_multi(A: _Array, s: int, seed: int, deterministic: bool = False) -> _Out:
  """Multiple-solution breadth-first search target generation."""
  rng = np.random.RandomState(seed)
  chex.assert_rank(A, 2)
  probeslist = []
  pies = []

  num_solutions = 1 if deterministic else _NUM_SOLUTIONS
  for _ in range(num_solutions):
    probes = probing.initialize(specs.SPECS['bfs_multi'])
    A_pos = np.arange(A.shape[0])
    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            's': probing.mask_one(s, A.shape[0]),
            'A': np.copy(A),
            'adj': probing.graph(np.copy(A))
        })

    reach = np.zeros(A.shape[0])
    pi = np.arange(A.shape[0])
    reach[s] = 1
    while True:
      prev_reach = np.copy(reach)
      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'reach_h': np.copy(prev_reach),
              'pi_h': np.copy(pi)
          })

      n = A.shape[0]
      sources = np.where(prev_reach == 1)[0]
      if deterministic:
        shuffled_sources = sources
      else:
        shuffled_sources = np.copy(sources)
        rng.shuffle(shuffled_sources)

      for src in shuffled_sources:
        for j in range(n):
          if A[src, j] > 0:
            if pi[j] == j and j != s:
              pi[j] = src
            reach[j] = 1
      if np.all(reach == prev_reach):
        break

    probing.push(probes, specs.Stage.OUTPUT, next_probe={'pi': np.copy(pi)})
    probing.finalize(probes)
    pies.append(pi)
    probeslist.append(probes)

  parent_dist = _parent_distribution_from_trees(pies, A.shape[0])
  probeslist[0]['output']['node']['pi']['data'] = parent_dist
  return parent_dist, probeslist[0]


def bellman_ford_multi(
    A: _Array, s: int, seed: int, deterministic: bool = False) -> _Out:
  """Multiple-solution Bellman-Ford target generation."""
  rng = np.random.RandomState(seed)
  chex.assert_rank(A, 2)
  A_pos = np.arange(A.shape[0])
  probeslist = []
  pies = []

  num_solutions = 1 if deterministic else _NUM_SOLUTIONS
  for _ in range(num_solutions):
    probes = probing.initialize(specs.SPECS['bellman_ford_multi'])
    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            's': probing.mask_one(s, A.shape[0]),
            'A': np.copy(A),
            'adj': probing.graph(np.copy(A))
        })

    d = np.zeros(A.shape[0])
    pi = np.arange(A.shape[0])
    msk = np.zeros(A.shape[0])
    d[s] = 0
    msk[s] = 1

    shuffled1 = np.arange(1, A.shape[0])
    rng.shuffle(shuffled1)
    shuffled1 = np.concatenate(([0], shuffled1))
    shuffled2 = np.arange(A.shape[0])
    rng.shuffle(shuffled2)

    while True:
      prev_d = np.copy(d)
      prev_msk = np.copy(msk)
      probing.push(
          probes,
          specs.Stage.HINT,
          next_probe={
              'pi_h': np.copy(pi),
              'd': np.copy(prev_d),
              'msk': np.copy(prev_msk)
          })
      for u in shuffled1:
        for v in shuffled2:
          if prev_msk[u] == 1 and A[u, v] != 0:
            if msk[v] == 0 or prev_d[u] + A[u, v] < d[v]:
              d[v] = prev_d[u] + A[u, v]
              pi[v] = u
            msk[v] = 1
      if np.all(d == prev_d):
        break

    probing.push(probes, specs.Stage.OUTPUT, next_probe={'pi': np.copy(pi)})
    probing.finalize(probes)
    pies.append(pi)
    probeslist.append(probes)

  parent_dist = _parent_distribution_from_trees(pies, A.shape[0])
  probeslist[0]['output']['node']['pi']['data'] = parent_dist
  return parent_dist, probeslist[0]


__all__ = ("dfs_multi", "bfs_multi", "bellman_ford_multi")

