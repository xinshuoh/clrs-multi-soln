"""Reusable metrics for multi-solution extraction."""

from typing import List

import numpy as np


def accuracy(mask: List[bool]) -> float:
  return float(sum(mask)) / float(len(mask)) if mask else 0.0


def multisol_score(pred, truth) -> float:
  """Higher-is-better POINTER_DISTRIBUTION similarity score."""
  return np.mean(1.0 - np.abs(pred - truth))
