import graphlib as gl
import networkx as nx
import numpy as np
import chex
import copy
## f[0][0][1].data to get adjacency matrix from next(sampler) where sampler=test_samplers[0]

# cyclic graph to test
cyclic_adj = np.array([
    [0,1,0],
    [0,0,1],
    [1,0,0]
])
cyclic_pi = [2,0,1]

acyclic_adj = np.array([
    [0,1],
    [0,0]
])
acyclic_pi = [0, 0]

disconnect_adj = np.array([
    [0,1,0],
    [0,0,0],
    [0,0,0]
])
disconnect_pi = [0,0,2]

edge_to_zero_adj = np.array([
    [0,0],
    [1,0]
])
edge_to_zero_pi = [0,1]


def brute_force_dfsTree_validity(A, pi):
    """builds all valid trees, checks if pi is one of them"""
    trees = []
    unmarked = [i for i in range(len(A)-1, -1, -1)]  # better locality, doing reverse
    while unmarked != []:
        node = unmarked[-1]
        stack = [node]
        while stack != []:
            # check outgoing edges
            a = l.pop()
            # append next-node to each tree being built

        ## recursion??


def check_valid_dfsTree_new(A, pi):
    '''checks: acyclic, dangling, edge-validity, and valid-start'''
    '''
        inputs: Adjacency Matrix (A), purported parent forest (pi)
        returns: True iff (pi) is producible by run of DFS with random edge exploration but ordered restarts
    '''
    pi = copy.deepcopy(pi) # copy pi. make sure don't mess with logging

    # build adjacency matrix of DFS forest
    size = len(pi)  # n_vertices
    M = np.zeros((size, size))
    for ix in range(size):
        M[int(pi[ix]), ix] = 1

    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    F = nx.from_numpy_array(M, create_using=nx.DiGraph)

    # for each child of node 0 in G, identify all descendants (vertices reachable from child)
    # find the children

    children = []   # children of source (node 0)
    for i in range(size):
        if M[0, i] == 1:
            children.append(i)

    descendants_children_G = descendants_children(G, children)
    descendants_children_F = descendants_children(F, children)

    while G.number_of_nodes != 0:
        changed = False
        for child in children:
            if descendants_children_G[child] == descendants_children_F[child]:
                # A valid run of DFS can discover all these descendants, then backtrack to source
                # So, remove all these descendants from graph (you don't have to worry about them, this part of forest is ok)
                changed = True
                for desc in descendants_children_G[child]:
                    G.remove_node(desc)
                    F.remove_node(desc)
                    if desc in children:
                        children.remove(desc)
                children.remove(child)
                #breakpoint()
                # recompute descendants of first-layer children, with marked-descendants removed
                descendants_children_G = descendants_children(G, children)
                descendants_children_F = descendants_children(F, children)
        if not changed:
            return False
    return True

def descendants_children(G, children):
    '''
    Method to return the descendants of a list of vertices in a nx.DiGraph as a dictionary.
    '''
    size = G.number_of_nodes()
    #breakpoint()
    #print('G=',G.nodes())
    #print('kids=', children)
    #print('size=', size)
    descendant_dict = {}
    for child in children:
        descs = []
        for node_id in G.nodes():
            if node_id != child:    # cant be self-parent if you're already child of source
                if nx.has_path(G, child, node_id):
                    descs.append(node_id)
        descendant_dict[child] = descs
    return descendant_dict

def check_valid_dfsTree(np_input_array, pi):
    '''checks: acyclic, dangling, edge-validity, and valid-start'''
    '''
        inputs: Adjacency Matrix (np_input_array), purported parent tree (pi)
        returns: True iff (pi) is producible by run of DFS with random edge exploration but ordered restarts
    '''
    pi = copy.deepcopy(pi) # copy pi. make sure don't mess with logging
    if pi[0] == 0: # correct start-node
        if are_valid_edges_parents(np_input_array, pi):
            if are_valid_order_parents(np_input_array, pi): # self-loops not reachable by lower_ix. Other parents reachable by ...
                pi = replace_self_loops_with_minus1(pi)
                if is_acyclic(pi): # no funky hiding parents. Should be implied by lower-node reachability.
                    return True
                else:
                    pass
                    #print('cycle')
            else:
                pass
                #print('oo')
        else:
            pass
            #print('not edges')
    else:
        pass
        #print('wrong startnode')
    return False


def replace_self_loops_with_minus1(pi):
    for i in range(len(pi)):
        if pi[i] == i:
            pi[i] = -1
    return pi

def are_valid_edges_parents(np_input_array, pi):
    pi = np.array(pi).astype(int)
    for i in range(len(pi)): # for node in graph.
        parent = pi[i]
        if parent != i: # Not a restart, check edge parent->child. (assume restarts are always valid, check later).
            #breakpoint()
            try:
                if np_input_array[parent][i] == 0: # no edge parent -> child
                    return False
            except:
                # breakpoint()
                pass
    return True

def are_valid_order_parents(np_input_array, pi):
    """
    Checks whether self-loops stem from valid DFS execution.
        If you were reachable_by_earlier_node, but have self-loop, it's a problem
    :param input:
    :param pi:
    :return:
    """
    g = nx.from_numpy_array(np_input_array, create_using=nx.DiGraph)
    for i in range(len(pi)):
        if pi[i] == i:
            for j in range(i): # crucially, this does not run when i=0, so the starting 0,0 self-loop always permitted
                self_reachable_by_earlier_node = nx.has_path(g, j, i) # does this do it directed? lower-triangle?
                if self_reachable_by_earlier_node:
                    return False
        else:  # not self-loop, so not a restart, make sure parents are reachable by lowest ix node from which i is reachable
            for j in range(i):
                if nx.has_path(g, j, i):  # lowest_ix node from which i is reachable
                    if not nx.has_path(g, j, pi[i]):  # parent not reachable. Bad! (note nx considers each node as having a path to itself)
                        return False
                    else:
                        break  # this parent is ok! advance to outermost for loop
    return True



def is_acyclic(pi):
    """
    Function to check for cycles in a predecessor array returned by the model
    :param input: the adjacency matrix of the graph on which the model has inferred the predecessor array
    :param pi: the predecessor array
    :return: Boolean indicating acyclicity
    """
    ts = gl.TopologicalSorter()
    for i in range(len(pi)):
        ts.add(i, pi[i])
    try:
        ts.prepare()
        return True
    except ValueError as e:
        if isinstance(e, gl.CycleError):
            #print("I am a cycle error")
            return False
        else:
            raise e


#print(is_acyclic(acyclic_adj, acyclic_pi))
#print(is_acyclic(cyclic_adj, cyclic_pi))

#print(is_acyclic(disconnect_adj, disconnect_pi))



##################################################################################################################
# BELLMAN-FORD CHECKER
##################################################################################################################
# Test whether model's pi represents valid bellman-ford



def bellman_ford_cost(A, s):
  """Bellman-Ford's single-source shortest path (Bellman, 1958).
  This has been taken and adapted from the original CLRS Bellman-Ford implementation"""


  chex.assert_rank(A, 2)

  A_pos = np.arange(A.shape[0])

  d = np.zeros(A.shape[0])
  pi = np.arange(A.shape[0])
  msk = np.zeros(A.shape[0])
  d[s] = 0
  msk[s] = 1
  while True:
    prev_d = np.copy(d)
    prev_msk = np.copy(msk)
    for u in range(A.shape[0]):
      for v in range(A.shape[0]):
        if prev_msk[u] == 1 and A[u, v] != 0:
          if msk[v] == 0 or prev_d[u] + A[u, v] < d[v]:
            d[v] = prev_d[u] + A[u, v]
            pi[v] = u
          msk[v] = 1
    if np.all(d == prev_d):
      break
  return d

def check_valid_BFpaths(A,s, parentpath):
    ''' 1. Computes true shortest path costs,
        2. Builds BF parent-tree subgraph of graph, computes shortest path costs on subgraph (model's paths)
    Note self-parent (pi[t] = t) is the default for no path s->t. '''

    true_costs = bellman_ford_cost(A,s)
    parentpath = np.array(parentpath).astype(int)

    # the adjacency matrix of the BF tree
    BF_tree_adj = np.zeros((len(parentpath),len(parentpath)))
    for i in range(len(parentpath)):
        #if parentpath[i] == i and i != s: # shortest path. forbids neg. weight cycle solutions
        #    return False
        #breakpoint()
        if A[parentpath[i],i] == 0 and parentpath[i] != i: # you're allowed to pick self-parents for unreachable nodes
            return False
        else:
            BF_tree_adj[parentpath[i],i] = A[parentpath[i],i]

    model_costs = bellman_ford_cost(BF_tree_adj, s)

    if (true_costs == model_costs).all():
        return True
    else:
        return False

test_graph = np.array([[0,1,1],
              [0,0,1],
              [0,0,0]])
s = 0
true_parentpath = [0,0,0]
false_parentpath = [0,0,1]

assert check_valid_BFpaths(test_graph,s,true_parentpath)
assert not check_valid_BFpaths(test_graph,s,false_parentpath)

d = np.array([
    [0., 0.],
    [0., 0.]])

s = 0

expect_d = [0,1]
assert check_valid_BFpaths(d,s,expect_d)


##################################################################################################################
# BFS CHECKER
##################################################################################################################

def check_valid_bfsTree(A, pi, s):
    pi = copy.deepcopy(pi)  # copy pi to avoid mutation

    # pi should have same length as number of nodes in A
    assert len(pi) == A.shape[0], "pi length must match number of nodes in A"
    pi = np.asarray(pi).astype(int)
    n = A.shape[0]

    if np.any(pi < 0) or np.any(pi >= n):
        return False

    # Compute shortest path distances from s in A.
    # Use directed graph semantics to match BFS implementations that test A[u, v].
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    dist = nx.single_source_shortest_path_length(G, s)

    # Source node must be its own parent
    if pi[s] != s:
        return False

    # Unreachable nodes must have self-parent.
    # Reachable nodes have valid parents and correct levels.
    for i in range(n):
        if i == s:
            continue
        if i not in dist:
            if pi[i] != i:
                return False
        else:
            parent = int(pi[i])
            if parent == i:
                return False
            if A[parent, i] == 0:
                return False
            if parent not in dist or dist[parent] != dist[i] - 1:
                return False

    # Enforce bfs_multi frontier-order consistency:
    # At each level, there must exist a single source order over previous-level
    # nodes that makes every chosen parent the first eligible source for its child.
    levels = {}
    for node, lvl in dist.items():
        levels.setdefault(lvl, []).append(node)

    max_level = max(levels.keys(), default=0)
    for lvl in range(1, max_level + 1):
        prev_level = levels.get(lvl - 1, [])
        cur_level = levels.get(lvl, [])
        if not cur_level:
            continue

        ordering_constraints = nx.DiGraph()
        ordering_constraints.add_nodes_from(prev_level)

        for child in cur_level:
            parent = int(pi[child])
            candidate_parents = [u for u in prev_level if A[u, child] != 0]
            if parent not in candidate_parents:
                return False
            for other in candidate_parents:
                if other != parent:
                    ordering_constraints.add_edge(parent, other)

        if not nx.is_directed_acyclic_graph(ordering_constraints):
            return False

    return True
