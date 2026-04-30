"""Multi-solution DFS target generation."""

from __future__ import annotations

from typing import Tuple

import chex
import numpy as np

from clrs._src import probing
from clrs._src import specs
from clrs._src.multi_sol.algorithms import common

_Array = np.ndarray
_Out = Tuple[_Array, probing.ProbesDict]


def dfs_multi(A: _Array, seed: int, deterministic: bool = False) -> _Out:
  """Multiple-solution depth-first search target generation."""
  chex.assert_rank(A, 2)
  return common.generate_parent_distribution_target(
      algorithm_name="dfs_multi",
      num_nodes=A.shape[0],
      seed=seed,
      deterministic=deterministic,
      run_single=lambda rng, algorithm_spec, _deterministic: _dfs_execution(
          A, rng, algorithm_spec),
  )


def _dfs_execution(A, rng, algorithm_spec):
  probes = probing.initialize(algorithm_spec)

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
  return pi, probes
