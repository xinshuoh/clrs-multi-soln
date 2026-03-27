"""Seed helpers for reproducible multi-solution sampling."""

import numpy as np


def make_rng(seed: int) -> np.random.RandomState:
  return np.random.RandomState(seed)

