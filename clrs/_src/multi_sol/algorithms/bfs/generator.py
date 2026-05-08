"""Multi-solution BFS target generation."""

from __future__ import annotations

from typing import Tuple

import chex
import numpy as np

from clrs._src import probing
from clrs._src import specs
from clrs._src.multi_sol.algorithms.common import generator_utils

_Array = np.ndarray
_Out = Tuple[_Array, probing.ProbesDict]


def bfs_multi(A: _Array, s: int, seed: int, deterministic: bool = False) -> _Out:
  """Multiple-solution breadth-first search target generation."""
  chex.assert_rank(A, 2)
  return generator_utils.generate_parent_distribution_target(
      algorithm_name="bfs_multi",
      num_nodes=A.shape[0],
      seed=seed,
      deterministic=deterministic,
      run_single=lambda rng, algorithm_spec, deterministic: _bfs_execution(
          A, s, rng, algorithm_spec, deterministic),
  )


def sample_solution(A: _Array, s: int, rng, deterministic: bool = False) -> _Array:
  """Sample a single BFS parent tree using generator-owned logic."""
  algorithm_spec = generator_utils.resolve_multisol_spec("bfs_multi")
  parent_tree, _ = _bfs_execution(A, s, rng, algorithm_spec, deterministic)
  return parent_tree


def _bfs_execution(A, s, rng, algorithm_spec, deterministic):
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
  return pi, probes
