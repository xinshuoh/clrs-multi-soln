"""MST-Prim multi-solution package."""

__all__ = ("mst_prim_multi")


def __getattr__(name):
  if name == "mst_prim_multi":
    from clrs._src.multi_sol.algorithms.mst_prim.generator import mst_prim_multi
    return mst_prim_multi
  raise AttributeError(name)
