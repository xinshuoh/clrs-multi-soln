"""MST-Prim validators."""

from clrs._src.multi_sol.validation import check_graphs


def check_valid_mst_prim_tree(adjacency, parent_tree, source):
  return check_graphs.check_valid_mstPrimTree(adjacency, parent_tree, source)
