import numpy as np
from clrs._src.multi_sol.sampling import dfs as dfs_sampling
from clrs._src.multi_sol.validation import dfs as dfs_validation


def check_uniqueness_dfs(As, probMatrices, n_samples = 5, method = "upwards"):
    uniques = []
    valids_uniques = []
    valids = []

    # issue: we are getting back list of trees from sample upwards, so we need to merge those in a way where
    # every index corresponds to the probmatrix we are sampling from
    #breakpoint()
    if method == "upwards":
        samples = np.array([dfs_sampling.sample_upwards(probMatrices) for j in range(n_samples)])
    elif method == "altupwards":
        samples = np.array([dfs_sampling.sample_altUpwards(probMatrices) for j in range(n_samples)])
    else:
        raise ValueError("Invalid Upwards Sampling method")
    probMatrices_format = dfs_sampling.extract_probMatrices(probMatrices)
    for i in range(len(probMatrices_format)):

        samples_matrix_i = samples[:,i] # first tree from each run, corresponds to n_samples runs on first graph

        # code from here: https://stackoverflow.com/questions/26514179/set-of-list-of-lists-in-python
        unique_trees = [list(item) for item in set([tuple(row) for row in samples_matrix_i])]

        #breakpoint()
        # save the fraction of unique samples
        uniques.append(len(unique_trees)/n_samples)

        valid_trees_of_uniques = [dfs_validation.check_valid_dfsTree(As[i],j) for j in unique_trees]
        valids_uniques.append(sum(valid_trees_of_uniques)/len(unique_trees))

        valid_trees= [dfs_validation.check_valid_dfsTree(As[i], j) for j in samples_matrix_i]
        valids.append(sum(valid_trees) / n_samples)
        #breakpoint()

    return uniques,valids_uniques, valids


'''
def check_uniqueness_bf(As, Ss, probMatrices, n_samples = 5, method = "beam"):
    uniques = []
    valids_uniques = []
    valids = []

    # issue: we are getting back list of trees from sample upwards, so we need to merge those in a way where
    # every index corresponds to the probmatrix we are sampling from
    #breakpoint()
    if method == "beam":
        samples = np.array([sample_beamsearch(As, Ss, probMatrices) for j in range(n_samples)])
    elif method == "rand":
        samples = np.array([dfs_sampling.sample_random_list([probMatrices]) for j in range(n_samples)])
    #elif method == "grdy":
    #elif method == "unif":
    else:
        raise ValueError("Invalid Upwards Sampling method")
    probMatrices_format = dfs_sampling.extract_probMatrices(probMatrices)
    for i in range(len(probMatrices_format)):

        samples_matrix_i = samples[:,i]

        # code from here: https://stackoverflow.com/questions/26514179/set-of-list-of-lists-in-python
        unique_trees = [list(item) for item in set([tuple(row) for row in samples_matrix_i])]

        breakpoint()
        # save the fraction of unique samples
        uniques.append(len(unique_trees)/n_samples)

        valid_trees_of_uniques = [dfs_validation.check_valid_dfsTree(probMatrices_format[i],j) for j in unique_trees]
        valids_uniques.append(sum(valid_trees_of_uniques)/len(unique_trees))

        valid_trees= [dfs_validation.check_valid_dfsTree(probMatrices_format[i], j) for j in samples_matrix_i]
        valids.append(sum(valid_trees) / n_samples)
        #breakpoint()

    return uniques,valids_uniques, valids'''


if __name__ == '__main__':
    class toy:
        def __init__(self, data):
            self.data = data

    A = np.array([
        [0.0, 0.5, 1.0],
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.0]
    ])

    pm = toy(
        np.array([
        [1,0,0],
        [1,0,0],
        [0,0.5,0.5]
        ])
    )

    d = np.array([
        [0,0],
        [0,0]
    ])
    dpm = toy(
        np.array([
            [1,0],
            [0,1]
        ])
    )

    check_uniqueness_bf([A, d], [0,0], [pm, dpm]) # extract probMatrices behaves weird with this test case
