import numpy as np
from clrs._src.multi_sol.sampling import bellman_ford as bf_sampling
from clrs._src.multi_sol.validation import check_graphs

def check_uniqueness_bf(probMatrices, source_nodes, As, n_samples = 5, method = "beam",values = "model"):
    uniques = []
    valids_uniques = []
    valids = []

    # issue: we are getting back list of trees from sample upwards, so we need to merge those in a way where
    # every index corresponds to the probmatrix we are sampling from
    #breakpoint()
    if method == "beam":
        samples = np.array([bf_sampling.sample_beamsearch(As, source_nodes,probMatrices) for j in range(n_samples)])
    elif method == "greedy":
        samples = np.array([bf_sampling.sample_greedysearch(As, source_nodes,probMatrices)for j in range(n_samples)])
    else:
        raise ValueError("Invalid Sampling method")
    #probMatrices_format = dfs_sampling.extract_probMatrices(probMatrices)
    for i in range(len(As)):

        samples_matrix_i = samples[:,i]

        # code from here: https://stackoverflow.com/questions/26514179/set-of-list-of-lists-in-python
        unique_trees = [list(item) for item in set([tuple(row) for row in samples_matrix_i])]


        # save the fraction of unique samples
        uniques.append(len(unique_trees)/n_samples)

        valid_trees_of_uniques = [check_graphs.check_valid_BFpaths(As[i],source_nodes[i],j) for j in unique_trees]
        valids_uniques.append(sum(valid_trees_of_uniques)/len(unique_trees))

        valid_trees= [check_graphs.check_valid_BFpaths(As[i],source_nodes[i],j) for j in samples_matrix_i]
        valids.append(sum(valid_trees) / n_samples)
        #breakpoint()

    return uniques,valids_uniques, valids
