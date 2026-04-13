"""Evaluation policy registry for extension output types.

Use `specs.Type.POINTER_DISTRIBUTION` for parent-distribution evaluation.
"""

from __future__ import annotations

from typing import Callable, Dict

from clrs._src import specs
from clrs._src.multi_sol.evaluation import metrics


_TYPE_EVAL_FNS: Dict[str, Callable] = {
    specs.Type.POINTER_DISTRIBUTION: metrics.multisol_score,
}


def registered_type_evals() -> Dict[str, Callable]:
  """Return a copy of extension type evaluators."""
  return dict(_TYPE_EVAL_FNS)
