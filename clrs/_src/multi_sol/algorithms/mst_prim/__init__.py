"""MST-Prim multi-solution package."""

__all__ = ("mst_prim_multi", "evaluate_mst_prim_multisol_batch")


def __getattr__(name):
  if name == "mst_prim_multi":
    from clrs._src.multi_sol.algorithms.mst_prim.generator import mst_prim_multi
    return mst_prim_multi
  if name == "evaluate_mst_prim_multisol_batch":
    from clrs._src.multi_sol.algorithms.mst_prim.plugin import (
        evaluate_mst_prim_multisol_batch,
    )
    return evaluate_mst_prim_multisol_batch
  raise AttributeError(name)
