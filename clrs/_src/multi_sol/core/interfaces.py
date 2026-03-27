"""Interfaces for extractor and validator plugins."""

from __future__ import annotations

import abc
from typing import List

import numpy as np


class Extractor(abc.ABC):
  """Samples parent trees from probability-matrix outputs."""

  @abc.abstractmethod
  def sample(self, outs_or_preds, source_nodes) -> List[np.ndarray]:
    """Return one sampled parent-tree per graph in batch."""


class Validator(abc.ABC):
  """Checks algorithm-specific validity for sampled parent trees."""

  @abc.abstractmethod
  def validate(self, adjacency: np.ndarray, parent_tree: np.ndarray,
               source: int) -> bool:
    """Return whether parent_tree is valid for given problem instance."""

