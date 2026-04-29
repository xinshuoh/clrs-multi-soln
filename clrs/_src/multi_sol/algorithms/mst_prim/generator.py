"""Multi-solution MST-Prim target generation."""

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


def mst_prim_multi(A: _Array, s: int, seed: int, deterministic: bool = False) -> _Out:
  """Multiple-solution target generation for Prim's MST algorithm."""
  chex.assert_rank(A, 2)
  return common.generate_parent_distribution_target(
      algorithm_name="mst_prim_multi",
      num_nodes=A.shape[0],
      seed=seed,
      deterministic=deterministic,
      run_single=lambda rng, algorithm_spec, deterministic: (
          _mst_prim_execution(A, s, rng, algorithm_spec, deterministic)),
      num_solutions=_NUM_SOLUTIONS,
  )


def _mst_prim_execution(A, s, rng, algorithm_spec, deterministic):
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

  key = np.zeros(A.shape[0])
  mark = np.zeros(A.shape[0])
  in_queue = np.zeros(A.shape[0])
  pi = np.arange(A.shape[0])
  key[s] = 0
  in_queue[s] = 1

  probing.push(
      probes,
      specs.Stage.HINT,
      next_probe={
          'pi_h': np.copy(pi),
          'key': np.copy(key),
          'mark': np.copy(mark),
          'in_queue': np.copy(in_queue),
          'u': probing.mask_one(s, A.shape[0])
      })

  for _ in range(A.shape[0]):
    effective_keys = np.where(in_queue == 1, key, np.inf)
    min_key_val = np.min(effective_keys)
    candidates = np.where(effective_keys == min_key_val)[0]
    u = candidates[0] if deterministic else rng.choice(candidates)
    if in_queue[u] == 0:
      break
    mark[u] = 1
    in_queue[u] = 0
    for v in range(A.shape[0]):
      if A[u, v] != 0:
        if mark[v] == 0 and (in_queue[v] == 0 or A[u, v] < key[v]):
          pi[v] = u
          key[v] = A[u, v]
          in_queue[v] = 1

    probing.push(
        probes,
        specs.Stage.HINT,
        next_probe={
            'pi_h': np.copy(pi),
            'key': np.copy(key),
            'mark': np.copy(mark),
            'in_queue': np.copy(in_queue),
            'u': probing.mask_one(u, A.shape[0])
        })

  probing.push(probes, specs.Stage.OUTPUT, next_probe={'pi': np.copy(pi)})
  probing.finalize(probes)
  return pi, probes
