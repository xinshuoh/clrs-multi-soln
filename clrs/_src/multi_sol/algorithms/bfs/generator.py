"""Multi-solution BFS target generation."""

from __future__ import annotations

from typing import Tuple

import chex
import numpy as np

from clrs._src import probing
from clrs._src import specs
from clrs._src.multi_sol.algorithms import common

_Array = np.ndarray
_Out = Tuple[_Array, probing.ProbesDict]
_NUM_SOLUTIONS = 20


def bfs_multi(A: _Array, s: int, seed: int, deterministic: bool = False) -> _Out:
  """Multiple-solution breadth-first search target generation."""
  rng = np.random.RandomState(seed)
  chex.assert_rank(A, 2)
  probeslist = []
  pies = []
  algorithm_spec = common.resolve_multisol_spec("bfs_multi")

  num_solutions = 1 if deterministic else _NUM_SOLUTIONS
  for _ in range(num_solutions):
    probes = probing.initialize(algorithm_spec)
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

  parent_dist = common.parent_distribution_from_trees(pies, A.shape[0])
  probeslist[0]['output']['node']['pi']['data'] = parent_dist
  return parent_dist, probeslist[0]

