"""MST-Prim multi-solution package."""

from clrs._src.multi_sol.algorithms.mst_prim.generator import mst_prim_multi
from clrs._src.multi_sol.algorithms.mst_prim.plugin import evaluate_mst_prim_multisol_batch

__all__ = ("mst_prim_multi", "evaluate_mst_prim_multisol_batch")
