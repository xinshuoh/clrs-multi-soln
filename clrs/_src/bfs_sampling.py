import numpy as np
from clrs._src import dfs_sampling

def sample_bfs_prim(outsOrPreds, s_indices):
    """
    Samples BFS trees using a Prim-like greedy approach.
    
    Args:
        outsOrPreds: Model outputs containing parent probabilities.
        s_indices: List of source node indices for each graph in the batch.
        
    Returns:
        List of parent arrays (pi), one for each graph in the batch.
    """
    probMatrix_list = dfs_sampling.extract_probMatrices(outsOrPreds)
    pi_trees = []
    
    # Handle case where s_indices might be a single int or list
    if isinstance(s_indices, int):
        s_indices = [s_indices] * len(probMatrix_list)
        
    for i, probMatrix in enumerate(probMatrix_list):
        s = s_indices[i]
        pi = prim_like_sampler(probMatrix, s)
        pi_trees.append(pi)
        
    return pi_trees

def prim_like_sampler(probMatrix, s):
    """
    Constructs a BFS tree starting from the root, adding nodes that are most likely 
    connected to the current tree.
    
    Args:
        probMatrix: N x N matrix where P[i, j] is prob that j is parent of i.
        s: Source node index.
        
    Returns:
        pi: Array of length N, where pi[i] is the parent of i.
    """
    num_nodes = probMatrix.shape[0]
    pi = np.full(num_nodes, -1, dtype=int) # -1 indicates not yet processed/no parent
    pi[s] = s
    
    processed = {s}
    remaining = set(range(num_nodes)) - {s}
    
    # To optimize, we could maintain scores, but N is likely small (<= 64 usually in CLRS)
    # So recomputing sum might be fast enough.
    
    while remaining:
        best_v = -1
        best_score = -1.0
        
        # Find v in remaining with highest prob of attaching to processed set
        # score(v) = sum_{u in processed} P(parent(v) = u)
        for v in remaining:
            # probMatrix[v, u] is prob that u is parent of v.
            # We sum over u in processed.
            score = sum(probMatrix[v, u] for u in processed)
            
            if score > best_score:
                best_score = score
                best_v = v
        
        if best_v == -1:
            # This happens if remaining nodes have 0 prob of connecting to processed nodes
            # (disconnected graph components). 
            # We just pick one arbitrarily or handle it?
            # For BFS, disconnected nodes typically point to themselves or have 0 parent in some representations.
            # Here, let's just pick the first one and attach it to itself (or closest processed if any non-zero).
            best_v = list(remaining)[0]
            # If score is 0, it means no connection.
            # We'll just process it to avoid infinite loop.
            
        # Sample parent for best_v from processed set
        # Probabilities are P(parent(best_v) = u) for u in processed
        # We need to normalize these to sum to 1.
        
        candidate_parents = list(processed)
        probs = np.array([probMatrix[best_v, u] for u in candidate_parents])
        
        total_prob = probs.sum()
        if total_prob > 0:
            probs = probs / total_prob
            parent = np.random.choice(candidate_parents, p=probs)
        else:
            # No valid parent in processed set (disconnected). 
            # Default to self-loop or some fallback.
            parent = best_v 
            
        pi[best_v] = parent
        processed.add(best_v)
        remaining.remove(best_v)
        
    return pi

def sample_bfs_categorical(outsOrPreds):
    """
    Samples parent pointers independently for each node using categorical sampling.
    Note: This does NOT guarantee a tree structure (may contain cycles or orphans).
    
    Args:           
        outsOrPreds: Model outputs containing parent probabilities.
        
    Returns:
        List of parent arrays (pi).
    """
    probMatrix_list = dfs_sampling.extract_probMatrices(outsOrPreds)
    pi_trees = []
    
    for probMatrix in probMatrix_list:
        num_nodes = probMatrix.shape[0]
        pi = np.zeros(num_nodes, dtype=int)
        
        for i in range(num_nodes):
            # Sample parent for node i based on the distribution in row i
            probs = probMatrix[i]
            # Ensure probabilities sum to 1 for np.random.choice
            total_p = np.sum(probs)
            assert total_p >= 0, "Probabilities must be non-negative"
            if total_p > 0:
                normalized_probs = probs / total_p
                pi[i] = np.random.choice(num_nodes, p=normalized_probs)
                
        pi_trees.append(pi)
        
    return pi_trees

def sample_bfs_beam(outsOrPreds, s_indices, beam_width=3):
    """
    Samples BFS trees using a Beam Search approach.
    
    Args:
        outsOrPreds: Model outputs containing parent probabilities.
        s_indices: List of source node indices.
        beam_width: Number of hypotheses to keep.
        
    Returns:
        List of parent arrays (pi).
    """
    probMatrix_list = dfs_sampling.extract_probMatrices(outsOrPreds)
    pi_trees = []
    
    if isinstance(s_indices, int):
        s_indices = [s_indices] * len(probMatrix_list)
        
    for i, probMatrix in enumerate(probMatrix_list):
        s = s_indices[i]
        pi = bfs_beam_sampler(probMatrix, s, beam_width)
        pi_trees.append(pi)
        
    return pi_trees

def bfs_beam_sampler(probMatrix, s, beam_width):
    num_nodes = probMatrix.shape[0]
    
    # State: (log_prob, pi_array, processed_set_as_tuple)
    # processed_set is stored as tuple for hashing/copying if needed, though sets are fine if we copy.
    # pi_array initialized with -1
    initial_pi = np.full(num_nodes, -1, dtype=int)
    initial_pi[s] = s
    
    # Beam: list of dicts or objects. 
    # Let's use a simple dict: {'log_prob': 0.0, 'pi': pi, 'processed': {s}}
    beam = [{'log_prob': 0.0, 'pi': initial_pi, 'processed': {s}}]
    
    # We need to add N-1 nodes
    for _ in range(num_nodes - 1):
        candidates = []
        
        for hyp in beam:
            processed = hyp['processed']
            pi = hyp['pi']
            curr_log_prob = hyp['log_prob']
            remaining = set(range(num_nodes)) - processed
            
            if not remaining:
                candidates.append(hyp)
                continue
            
            # 1. Select next node `v` to add (Greedy heuristic)
            # We pick the v with highest total probability of connecting to current processed set
            best_v = -1
            best_connection_mass = -1.0
            
            for v in remaining:
                # Sum of probs from processed nodes to v
                # probMatrix[v, u] is P(parent(v)=u)
                mass = sum(probMatrix[v, u] for u in processed)
                if mass > best_connection_mass:
                    best_connection_mass = mass
                    best_v = v
            
            if best_v == -1:
                best_v = list(remaining)[0] # Disconnected/fallback
                
            # 2. Branch on all possible parents u in processed for best_v
            possible_parents = list(processed)
            
            for u in possible_parents:
                p_val = probMatrix[best_v, u]
                if p_val > 1e-9: # Avoid log(0)
                    new_log_prob = curr_log_prob + np.log(p_val)
                else:
                    new_log_prob = curr_log_prob - 1e9 # Penalty
                
                new_pi = pi.copy()
                new_pi[best_v] = u
                new_processed = processed.copy()
                new_processed.add(best_v)
                
                candidates.append({
                    'log_prob': new_log_prob,
                    'pi': new_pi,
                    'processed': new_processed
                })
        
        if not candidates:
            break # Should not happen usually unless disconnected

        # Select top-k candidates
        # Sort by log_prob descending
        candidates.sort(key=lambda x: x['log_prob'], reverse=True)
        beam = candidates[:beam_width]
        
    return beam[0]['pi'] # Return best hypothesis