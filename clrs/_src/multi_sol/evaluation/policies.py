"""Evaluation policy registry for extension output types.

Prefer `specs.Type.MULTI_SOLUTION` for new registrations. The legacy
`specs.Type.MULT_SOL` key is kept as a temporary compatibility alias.
"""

from __future__ import annotations

from typing import Callable, Dict

from clrs._src import specs
from clrs._src.multi_sol.evaluation import metrics


_TYPE_EVAL_FNS: Dict[str, Callable] = {
    specs.Type.MULTI_SOLUTION: metrics.multisol_score,
}


def registered_type_evals() -> Dict[str, Callable]:
  """Return a copy of extension type evaluators."""
  return dict(_TYPE_EVAL_FNS)
