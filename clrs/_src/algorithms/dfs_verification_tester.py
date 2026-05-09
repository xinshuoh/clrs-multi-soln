# Jan 27
# implement algorithms for testing DFS-verification to catch any slip-ups with code

# generate random graphs
# dfs those graphs to create forests
# run verification algorithms on forests to ensure all real forests are marked good

# create deliberately bad forests? or just random forests?
# run verification algorithms to ensure fake forests are marked bad
#   **there's a chance you randomly make a good forest, save those for manual investigation**

import networkx as nx  # graph library
import matplotlib.pyplot as plt  # visualize nx graphs
import random  # shuffle lists
import numpy as np  # adjacency matrices

# ---------------------------------------------------------------------------------------------------------------------
# -- Generating Random Graphs -- #
# ---------------------------------------------------------------------------------------------------------------------
random.seed(123)
NUM_TO_TEST = 10**3


def er_graph(n, p):
  # Generate random graph
  G = nx.erdos_renyi_graph(n, p, directed=True)
  return G


def random_subgraphs(G, p, n):
  return [random_subgraph(G, p) for i in range(n)]


def random_subgraph(G, p):
  G = nx.DiGraph(G)
  subgraph = nx.DiGraph()
  subgraph.add_nodes_from(G.nodes())
  # keep each edge with probability p
  random_edges = []
  for edge in G.edges():
    if random.random() < p:
      random_edges.append(edge)
  # Create a new subgraph with only these edges
  subgraph.add_edges_from(random_edges)
  return subgraph


def random_forestish():
  # ensure no cycles, make it a forest
  return


def random_forest_subgraph(graph, p):
  """Create a subgraph where edges are randomly removed, then extract spanning trees."""
  # Step 1: Keep each edge with probability p
  subgraph = nx.Graph() if not graph.is_directed() else nx.DiGraph()
  subgraph.add_nodes_from(graph.nodes())  # Keep all nodes

  for u, v in graph.edges():
    if random.random() < p:  # Keep edge with probability p
      subgraph.add_edge(u, v)

  # Step 2: Extract a spanning tree from each connected component
  forest = nx.Graph()  # A forest is an undirected graph with multiple trees
  forest.add_nodes_from(subgraph.nodes())  # Keep all nodes

  for component in nx.connected_components(subgraph):  # Find connected components
    tree = nx.minimum_spanning_tree(subgraph.subgraph(component))  # Get a spanning tree
    forest.add_edges_from(tree.edges())  # Add tree edges to the forest

  return forest


# ---------------------------------------------------------------------------------------------------------------------
# -- DFS with Ordered Restarts -- #
# ---------------------------------------------------------------------------------------------------------------------


def dfs(graph):
  visited = set()
  stack = []  # Start with the initial node
  traversal = []  # Store the traversal order
  all_nodes = {i for i in range(len(graph))}

  while len(visited) != len(graph):
    stack.append(min(all_nodes - visited))  # restart @ smallest unseen node
    while stack:
      node = stack.pop()  # Pop a node from the stack
      if node not in visited:
        visited.add(node)
        traversal.append(node)

        # Add neighbors to the stack in shuffle order
        nbrs = list(graph.neighbors(node))
        stack.extend(random.sample(
            nbrs, len(nbrs)))  # random.sample here shuffles nbrs for random edge explore

  return traversal


def forest_from_traversal(G, O):
  '''discoverer is closest preceding in order with edge?'''
  #print('assuming order and graph uniquely determine tree')
  forest = np.zeros((len(G), len(G)))
  back = list(reversed(O))
  for ix in range(len(back)):
    for jx in range(ix, len(back)):
      i = back[ix]
      j = back[jx]
      if (j, i) in G.edges:
        # i discovers j
        forest[j, i] = 1
        break
  return forest


# ---------------------------------------------------------------------------------------------------------------------
# -- Henry's O(n^4) -- #
# ---------------------------------------------------------------------------------------------------------------------
#
def parent_tree_to_adj_matrix(tree):
  size = len(tree)  # n_vertices
  M = np.zeros((size, size))
  for ix in range(size):
    M[int(tree[ix]), ix] = 1  # edge points tree[ix] to ix, bcuz parent tree
  return M


def no_self_loops_parent_tree_to_adj_matrix(
    tree):  # FIXME: duplicate code in validate_distributions cuz im lazy
  """now root is just any node without parent"""
  M = parent_tree_to_adj_matrix(tree)
  np.fill_diagonal(a=M, val=0)
  return M


def preprocess(adj_matrix):
  """ensure it's acyclic, nodes have at most 1 parent"""
  if isinstance(adj_matrix, nx.Graph):
    adj_matrix = nx.to_numpy_array(adj_matrix)
  n = len(adj_matrix)
  # Step 1: Count incoming edges for each node
  in_degrees = np.sum(adj_matrix, axis=0)
  # Step 2: Ensure nodes have no more than 1 parent
  if not np.all((in_degrees == 1) | (in_degrees == 0)):
    return False  # Each node must have either 0 (root) or 1 incoming edge
  # Step 3: Ensure acyclic
  F = nx.DiGraph(adj_matrix)

  return nx.is_forest(F)


def dfsverify(G, F):
  if isinstance(F, (list, np.ndarray)):  # check if its a parent tree format, convert
    if np.ndim(F) == 1:
      F = no_self_loops_parent_tree_to_adj_matrix(F)
  if not preprocess(F):
    return False
  if isinstance(G, np.ndarray):
    G = nx.from_numpy_array(G, create_using=nx.DiGraph)
  if isinstance(F, np.ndarray):
    F = nx.from_numpy_array(F, create_using=nx.DiGraph)
  colors = [0] * len(G)
  for i in range(len(G)):  # outer restart loop
    if not ccv(G, F, i, colors):
      #print(f'fails on node {i}')
      return False
  return True


def notgreen(kids, color):
  '''kids is a list of node ix'''
  return [kid for kid in kids if color[kid] != 1]


def down(G, source, colors):
  '''return set of all descendents of node in G, cannot pass through green nodes'''
  visited = set()
  descendants = set()

  def notgreen_dfs(node):
    if node in visited or colors[node] == 1:
      return
    visited.add(node)
    descendants.add(node)
    for neighbor in G.neighbors(node):
      notgreen_dfs(neighbor)

  notgreen_dfs(source)
  descendants.discard(
      source)  # Exclude the source itself. If source not present (green at start) does nothing
  return descendants


def treedown(G, source):
  '''return set of all descendents of node in G'''  # testing this in tree to deal with bug? PROBLEM, accepts disconnected bois
  visited = set()
  descendants = set()

  def minidfs(node):
    if node in visited:
      return
    visited.add(node)
    descendants.add(node)
    for neighbor in G.neighbors(node):
      minidfs(neighbor)

  minidfs(source)
  descendants.discard(source)  # Exclude the source itself.
  return descendants


def ccv(G, F, node, colors):
  '''check child validity: which kid can go next, like a waterslide'''  # FIXME: you can early-discover dead-ends, which can lead you to false rejecting bcuz not descendant in graph but yes in tree, fix by following tree-edges for kids, not graph-edges
  if colors[
      node] == 1:  # you can only be greened by your kids being valid at some point, and once you're green you stay green
    return True
  if down(G, node,
          colors) != treedown(F, node):  #down(F, node, colors):  # base case: this node can go
    return False
  colors[node] = 1  # passed the vibe check, green
  #print(f'greening node {node}')
  #kids = G.neighbors(node)
  kids = F.neighbors(node)
  kids = notgreen(kids, colors)  # green means visited, done
  i = 0
  while i < len(kids):
    kid = kids[i]
    pd = dict()
    ad = dict()
    pd[kid] = down(G, kid, colors)  # possible descendants **through non-green paths**
    ad[kid] = treedown(
        F,
        kid)  #down(F, kid, colors)  # actual descendants **should this be thru non-green? it is**
    #breakpoint()
    if pd[kid] == ad[kid]:
      # this kid can go next
      if not ccv(G, F, kid, colors):  # blow-up if things are bad in the descendants
        #print('fails on descendants')
        return False
      i = 0  # otherwise, proceed to other top-level kids
      kids = notgreen(kids, colors)  # change the loop list
    else:
      i += 1  # this one cant go first. check the next top-level kid

  return notgreen(kids, colors) == [
  ]  # all kids green we gucci, bcuz all subkids green or else return false early


# ---------------------------------------------------------------------------------------------------------------------
# -- Test on Known Graphs -- #
# ---------------------------------------------------------------------------------------------------------------------


def get_unique_adjacency_matrices(graphs):
  unique_matrices = set()
  unique_graphs = []
  #breakpoint()
  for G in graphs:
    if isinstance(G, np.ndarray):
      adj_matrix = G
    else:
      # Get the adjacency matrix
      adj_matrix = nx.to_numpy_array(G)

    # Convert matrix to a tuple of tuples (hashable format)
    matrix_tuple = tuple(map(tuple, adj_matrix))

    # Check uniqueness
    if matrix_tuple not in unique_matrices:
      unique_matrices.add(matrix_tuple)
      unique_graphs.append(G)

  return unique_graphs


def graphtest(G, verifier):
  false_approves = []
  false_rejects = []
  go, good_forests = prolly_unique_dfs(G)
  bad_graphs = random_subgraphs(G, p=0.5, n=NUM_TO_TEST)
  #breakpoint()
  #bad_forests = random_forestish(G)
  for g in good_forests.values():
    if not verifier(G, g):
      false_rejects.append(g)
  for b in bad_graphs:
    if verifier(G, b):
      false_approves.append(b)
  return get_unique_adjacency_matrices(false_approves), get_unique_adjacency_matrices(
      false_rejects), len(go)


# mbe back edges problem, extra edge problem**


def dfs_many_times(G, n_runs):
  traversal_orders = []
  trees = {}
  for i in range(n_runs):
    order = tuple(dfs(G))
    traversal_orders.append(order)
    trees[order] = forest_from_traversal(G, order)
  return traversal_orders, trees


def prolly_unique_dfs(G):
  #print('very approximate: the unique dfs trees of G. ok for small graphs (<= 6 nodes)')
  orders, trees = dfs_many_times(G, NUM_TO_TEST)
  unique_orders = set(orders)
  shortlist_trees = {order: trees[order] for order in unique_orders}
  return unique_orders, shortlist_trees


def draw(G):
  if isinstance(G, np.ndarray):
    G = nx.from_numpy_array(G, create_using=nx.DiGraph)
  nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500)
  plt.show()


# three-arm graph (0->2->5, 0->1->3->6, 0->1->4,->7), 4 possibilities (do 1 first or do 0 first, swap47and36)
# expect [0,2,5,1,4,7,3,6], [0,2,5,1,3,6,4,7], [0,1,3,6,4,7,2,5], [0,1,4,7,3,6,2,5],
# G = nx.Graph()
# #edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6), (4, 7)]   # connected
# edges = [(0, 1), (0, 2), (1, 4), (2, 5), (3, 6), (4, 7)]            # disconnected
# G.add_edges_from(edges)
# to = dfs(G)    # rerunning gives new dfs's
# F = nx.from_numpy_array(forest_from_traversal(G, to), create_using=nx.DiGraph)
#
# # CLASSIC TRIANGLE
# T = nx.Graph()
# tedges = [(0,1), (0,2), (1,2)]
# T.add_edges_from(tedges)
# tos, tts = prolly_unique_dfs(T)
#
# false = np.array([[0,1,1], [0,0,0], [0,0,0]])

#TESTING - "false acceptances" (gfa/tfa) should be visually correct trees when you draw them, orelse algo is wrong

# print('num false accepts g: ', len(gfa))
# print('num false accepts t: ', len(tfa))

#for g in gfa:
#    draw(g)
#
# for t in tfa:
#     draw(t)

# TRICKY EXAMPLE FOUND BY SANITY CHECK: problem - T is giving initial nodes new parents
A = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0], [1, 1, 0, 0, 0, 1],
              [0, 1, 1, 0, 0, 1], [0, 1, 0, 0, 0, 0]])

T = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1],
              [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])

A2 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
T2 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

A3 = np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]])
T3 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])

FRA = np.array([[0, 1, 0, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0, 1], [1, 0, 1, 1, 1, 0]])
FRT = np.array([[0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 0]])


def manual_sanity_check(graphsizes, verifier_algorithm):
  """
    for each size in graphsizes,
        1. make a random graph,
        2. run graphtest(G,verfier), which tests 1000 good trees and 1000 bad trees
        3. assert no false rejects (all good trees are marked good)
        4. store `probably false accepts` for manual inspection (bad trees are just random trees, they might be correct)
    """
  outcomes = []
  for size in graphsizes:
    for i in range(10):
      G = er_graph(n=size, p=0.6)  # slightly dense
      fa, fr, nu = graphtest(G, verifier_algorithm)
      if len(fr) != 0:
        print(f'Problems with verifier on size {size}')
      outcomes.append((G, fa, fr))
  return outcomes


def automatic_sanity_check(
    n=64,
    verifier_algorithm=dfsverify
):  # you should expect to randomly generate a correct tree on n=6 with 1000 random trees
  """
    1. make 10 random graphs of size n,
    2. create 1000 good trees and 1000 probably bad trees,
    3. assert no false rejects
    4. flag false accepts (should be very unlikely for sufficiently large n)
    """
  outcomes = []
  nums_unique_trees_found = []
  for i in range(10):
    G = er_graph(n=n, p=0.5)
    false_accepts, false_rejects, num_unique_true_trees_found = graphtest(G, verifier_algorithm)
    if len(false_rejects) != 0:
      print(f'Problems with verifier on Graph {i}')
    #if len(false_accepts) != 0:
    #    print(f'Manually inspect `false accept` on Graph {i}')
    outcomes.append((G, false_accepts, false_rejects))
    nums_unique_trees_found.append(num_unique_true_trees_found)
  how_many_fa(outcomes, nums_unique_trees_found)
  return outcomes


def how_many_fa(outcomes, nus):
  ix = 0
  for triple in outcomes:
    nu = nus[ix]
    G = triple[0]
    false_accepts = triple[1]
    false_rejects = triple[2]
    print(
        f'Graph {ix} has {len(false_accepts)} possibly false accepts and {len(false_rejects)} false rejects || for-context, we saw {nu} unique trees'
    )
    ix += 1


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

# -- O(n^3) bcuz computing reachability matrices takes 4ever -- #
# FALSE. ORDERING NOT EASY TO INFER
# idea: create matrix ordering by  rules
# topologically-sort resulting graph using DFS (does determinism matter here? what if tiebreaks weird?)
# verify that roots are in proper order (both by rules and lexicographic)
# the topological sort is new labeling
# run deterministic DFS, un-relabel, check identity.

#
# def reachability_floyd_warshall(G):
#     '''1 at row x col y, iff path x->y. else 0. Nodes are self-reachable'''
#     adj_matrix = nx.to_numpy_array(G)
#     n = len(adj_matrix)
#     reach = adj_matrix.copy()
#
#     for k in range(n):
#         for i in range(n):
#             for j in range(n):
#                 reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])
#
#     return reach
#
# def orderings_from_reach_mats(gmat, fmat):
#     '''ordering_mat is matrix, (i,j)=1 iff i<j based on reachability'''
#     ordering_mat = np.zeros((len(gmat), len(gmat)))
#     for source_ix in range(len(gmat)):
#         for target_ix in range(len(gmat)):
#             if gmat[source_ix, target_ix] and fmat[source_ix, target_ix]:
#                 # then source b4 target
#                 ordering_mat[source_ix, target_ix] = 1
#             elif gmat[source_ix, target_ix] and not fmat[source_ix, target_ix]:
#                 # target b4 source
#                 ordering_mat[target_ix, source_ix] = 1
#             elif not gmat[source_ix, target_ix] and not fmat[source_ix, target_ix]:
#                 # conclude nothing. not reachable
#                 pass
#             else:
#                 print('error, reachable in F but not G, illegal')
#     return ordering_mat - np.eye(len(gmat), len(gmat)) # remove self<self
#
#
# def Algo(G, F):
#     g_reach_matrix = reachability_floyd_warshall(G)
#     f_reach_matrix = reachability_floyd_warshall(F)
#     constraint_graph = orderings_from_reach_mats(g_reach_matrix, f_reach_matrix)
#     return constraint_graph
#     # toposort constraint graph for labelling
#
# #nx.to_numpy_array(G)
#
# A = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=int)
# reach_matrix = reachability_floyd_warshall(nx.from_numpy_array(A))
# #print(reach_matrix)
#outcomes = automatic_sanity_check(n=6)
