"""Multi-solution Bellman-Ford target generation."""

from __future__ import annotations

from typing import Tuple

import chex
import numpy as np

from clrs._src import probing
from clrs._src import specs
from clrs._src.multi_sol.algorithms import common

_Array = np.ndarray
_Out = Tuple[_Array, probing.ProbesDict]


def bellman_ford_multi(
    A: _Array, s: int, seed: int, deterministic: bool = False) -> _Out:
  """Multiple-solution Bellman-Ford target generation."""
  chex.assert_rank(A, 2)
  return common.generate_parent_distribution_target(
      algorithm_name="bellman_ford_multi",
      num_nodes=A.shape[0],
      seed=seed,
      deterministic=deterministic,
      run_single=lambda rng, algorithm_spec, deterministic: (
          _bellman_ford_execution(A, s, rng, algorithm_spec)),
  )


def _bellman_ford_execution(A, s, rng, algorithm_spec):
  A_pos = np.arange(A.shape[0])
  probes = probing.initialize(algorithm_spec)
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
  return pi, probes
