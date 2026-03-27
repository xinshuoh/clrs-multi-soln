"""Validation utility helpers."""

from typing import Callable, List


def validate_trees(trees, adjacency, sources, validate_fn: Callable):
  """Validate one sampled tree per graph."""
  return [
      validate_fn(adjacency[i], trees[i], int(sources[i]))
      for i in range(len(trees))
  ]


def accuracy(mask: List[bool]) -> float:
  return float(sum(mask)) / float(len(mask)) if mask else 0.0

