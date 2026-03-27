from clrs._src.algorithms.BF_beamsearch import BF_beamsearch, BF_greedysearch
from clrs._src.dfs_sampling import get_parent_tree_upwards, single_sample_upwards

from clrs._src.algorithms.check_graphs import check_valid_BFpaths, check_valid_dfsTree
from clrs._src.algorithms.dfs_verification_tester import dfsverify
from clrs._src.multi_sol.data.distribution import extract_probability_matrices
from sciplotlib import style as spstyle
from clrs._src.algorithms.graphs import bellman_ford, dfs

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print('graph1 hooks to BF_collect_and_eval in log_experiments.py')
print('edge_reuse_matrix_list works on dummy example')

# TODO: make make_edge_reuse_matrix_list split by unique/valid trees


# ----------------------------------------------------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------------------------------------------------
def parent_tree_to_adj_matrix(tree):
    size = len(tree)    # n_vertices
    M = np.zeros((size, size))
    for ix in range(size):
        M[int(tree[ix]), ix] = 1     # edge points tree[ix] to ix, bcuz parent tree
    return M

def no_self_loops_parent_tree_to_adj_matrix(tree):
    """now root is just any node without parent"""
    M = parent_tree_to_adj_matrix(tree)
    np.fill_diagonal(a=M, val=0)
    return M

def adj_matrix_to_parent_tree(A):
    size = len(A)
    tree = np.zeros(size)
    for i in range(size):
        for j in range(size):
            if A[i,j] == 1:
                tree[j] = i
    return tree

# ----------------------------------------------------------------------------------------------------------------------
# DO THE THING WITH LINE GRAPHS
# GIVEN, a distribution "preds", and an adjacency matrix
# 1. run sampling algorithms increasingly many times on preds
# 2. count how many valid unique solutions
# 3. count how many invalid unique solutions
# ----------------------------------------------------------------------------------------------------------------------

# solution, beamOrGreedy?, valid?, unique?, how_many_times_found?, (maybe also adjacency matrix)
# ----------------------------------------------------------------------------------------------------------------------
# MAIN FUNCTION... validate_distributions, calls BF or DFS depending on value of `flag`.
# ----------------------------------------------------------------------------------------------------------------------
def validate_distributions(As, Ss, outsOrPreds, flag, numSolsExtracting = 100, edge_reuse_BF= False, edge_reuse_DFS = False):
    #breakpoint()
    probMatrix_list = extract_probability_matrices(outsOrPreds)
    dataframes = []
    pMs = []
    for ix in range(len(probMatrix_list)):
        #breakpoint()
        A = As[ix]
        startNode = Ss[ix]
        probMatrix = probMatrix_list[ix]
        # build a plot,
        if flag=='BF':
            dataframes.append(make_n_unique_by_n_extracted_df(A=A, s=startNode, pred=probMatrix, num_solutions_extracted=numSolsExtracting))
        elif edge_reuse_BF:
            matrix_lists = make_edge_reuse_matrix_list(A, startNode, probMatrix,numSolsExtracting)
            df = postprocess_edge_reuse_matrix_list(matrix_lists)
            dataframes.append(df)
        elif flag=='DFS':
            df, A, pM = DFS_graph1_df(A=A, pred=probMatrix, num_solutions_extracted=numSolsExtracting)
            dataframes.append(df)
            pMs.append(pM)
        elif edge_reuse_DFS:
            matrix_lists = make_edge_reuse_matrix_list_dfs(A, probMatrix, numSolsExtracting)
            df = postprocess_edge_reuse_matrix_list_dfs(matrix_lists)
            dataframes.append(df)
        else:
            raise ValueError('no flag given to validate_distributions')
    return dataframes, As, pMs
# ----------------------------------------------------------------------------------------------------------------------
# DFS
# ----------------------------------------------------------------------------------------------------------------------
def DFS_graph1_df(A, pred, num_solutions_extracted):
    # columns for dataframe
    upwards_trees = []
    upwards_valid = []
    upwards_unique = []

    dfs_trees = []
    dfs_valid = []
    dfs_unique = []

    altUpwards_trees = []
    altUpwards_valid = []
    altUpwards_unique = []

    # frequency dicts
    upwards_dict = dict()
    altUpwards_dict = dict()
    dfs_dict = dict()

    # core routine
    num_sampled = 0
    while num_sampled < num_solutions_extracted:
        num_sampled += 1
        # extract new trees
        up_tree = single_sample_upwards(pred)
        alt_tree = get_parent_tree_upwards(pred)
        dfs_tree = adj_matrix_to_parent_tree(dfs(A, deterministic=True)[0])
        # save them for later dataframe
        upwards_trees.append(up_tree)
        altUpwards_trees.append(alt_tree)
        dfs_trees.append(dfs_tree)

        #breakpoint()
        # valid?
        upwards_valid.append(dfsverify(A, no_self_loops_parent_tree_to_adj_matrix(up_tree)))
        altUpwards_valid.append(dfsverify(A, no_self_loops_parent_tree_to_adj_matrix(alt_tree)))

        # unique?
        hash_up = hash(tuple(up_tree))
        upwards_unique.append(hash_up not in upwards_dict.keys())
        hash_alt = hash(tuple(alt_tree))
        altUpwards_unique.append(hash_alt not in altUpwards_dict.keys())
        hash_dfs = hash(tuple(dfs_tree))
        dfs_unique.append(hash_dfs not in dfs_dict.keys())

        # update freq dict
        if hash_up not in upwards_dict.keys():
            upwards_dict[hash_up] = 1
        else:
            upwards_dict[hash_up] += 1

        if hash_alt not in altUpwards_dict.keys():
            altUpwards_dict[hash_alt] = 1
        else:
            altUpwards_dict[hash_alt] += 1

        if hash_dfs not in dfs_dict.keys():
            dfs_dict[hash_dfs] = 1
        else:
            dfs_dict[hash_dfs] += 1

    df = pd.DataFrame.from_dict(
            {'up_trees': upwards_trees, 'unique_up': upwards_unique, 'valid_up': upwards_valid,
             'alt_trees': altUpwards_trees, 'unique_alt': altUpwards_unique, 'valid_alt': altUpwards_valid,
             'dfs_trees' : dfs_trees, 'unique_dfs': dfs_unique}
    )
    df['total_unique_seen'] = df['unique_alt'].cumsum()
    df['total_valid_seen'] = df['valid_alt'].cumsum()
    df['total_uv_seen_alt'] = (df['unique_alt'] & df['valid_alt']).cumsum()
    df['total_uv_seen_upwards'] = (df['unique_up'] & df['valid_up']).cumsum()
    df['total_uv_seen_dfs'] = df['unique_dfs'].cumsum()

    return df, A, pred


def plot_edge_reuse_matrix_list_mean_dfs(df, graphsize):
    """FIXME: .iloc[-1] is taking only the last number? after the max number of solutions."""
    with plt.style.context(spstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots(ncols=1, sharey=True)
    df = pd.concat(df)
    # u & v
    mean_edge_reuse_upwards_mean = df['upwards_means'].groupby(df.index).mean().iloc[-1]
    mean_edge_reuse_upwards_std = df['upwards_means'].groupby(df.index).std().iloc[-1]

    mean_edge_reuse_alt_mean = df['alt_means'].groupby(df.index).mean().iloc[-1]
    mean_edge_reuse_alt_std = df['alt_means'].groupby(df.index).std().iloc[-1]

    mean_edge_reuse_dfs_mean = df['dfs_means'].groupby(df.index).mean().iloc[-1]
    mean_edge_reuse_dfs_std = df['dfs_means'].groupby(df.index).std().iloc[-1]

    means = [mean_edge_reuse_upwards_mean, mean_edge_reuse_alt_mean, mean_edge_reuse_dfs_mean]
    std = [mean_edge_reuse_upwards_std, mean_edge_reuse_alt_std, mean_edge_reuse_dfs_std]

    plt.bar(np.arange(len(means)), means, 0.4)
    plt.errorbar(np.arange(len(means)), means, yerr=std, color="black", capsize=5, ls="None")
    plt.xticks(np.arange(len(means)), ["Upwards", "AltUpwards", "DFS"])

    # plt.plot([i for i in range(len(mean_edge_reuse_upwards_mean))], mean_edge_reuse_upwards_mean, marker='o', linestyle='-',color="blue", label="Upwardssearch")
    # plt.fill_between([i for i in range(len(mean_edge_reuse_upwards_mean))], mean_edge_reuse_upwards_mean - mean_edge_reuse_upwards_std,mean_edge_reuse_upwards_mean + mean_edge_reuse_upwards_std, color="blue", alpha=0.15)

    # plt.plot([i for i in range(len(mean_edge_reuse_upwards_mean))], mean_edge_reuse_alt_mean, marker='x', linestyle='-', color="red", label="altUpwards")
    # plt.fill_between([i for i in range(len(mean_edge_reuse_upwards_mean))],mean_edge_reuse_alt_mean - mean_edge_reuse_alt_std,mean_edge_reuse_alt_mean + mean_edge_reuse_alt_std, color="red", alpha=0.15)
    # plt.plot([i for i in range(len(total_uv_seen_upwards_mean))], total_uv_seen_alt_mean, marker='v', linestyle='-', color="green", label="DFS")
    # plt.plot([i for i in range(len(mean_edge_reuse_upwards_mean))], mean_edge_reuse_dfs_mean, marker='v', linestyle='-', color="green", label="DFS")
    # plt.fill_between([i for i in range(len(mean_edge_reuse_upwards_mean))],mean_edge_reuse_dfs_mean - mean_edge_reuse_dfs_std,mean_edge_reuse_dfs_mean + mean_edge_reuse_dfs_std, color="red", alpha=0.15)
    # plt.legend(loc="upper left")
    # plt.plot(df.n_samples, df.medians, marker='o', linestyle='-')
    # plt.axis((0, len(df), 0, 1))  # weird error, when I run in pycharm can't adjust axes, but works in terminal
    plt.title(f'Mean average edge reuse for n = {graphsize} (DFS)')
    plt.ylabel('Mean average edge reuse')
    plt.tight_layout()
    plt.savefig("edge_reuse_mean_" + str(graphsize) + "_dfs.png")
    plt.close()


def postprocess_edge_reuse_matrix_list_dfs(matrix_lists):
    """
    Convert many solutions (matrix_lists) from a single graph & predicted probMatrix, into a df of means by n_sols
    Args:
        matrix_lists: a list of lists [[alt_list], [upwards_list], [dfs_list]]
            each inner list (e.g. [alt_list]) contains adjacency matrices. Each adjacency matrix is a BF path.

    Returns:
        df: pandas Dataframe with alt, upwards, and dfs means and medians
    """
    # this is all for a single graph, where each matrix represents a subset of edges... perhaps in graph... extracted from parent tree
    # at each interval, sum adjacency matrices, calculate frequency, report score
    # TODO: filter by validity?

    # breakpoint()
    # n_samples_list = []
    medians = []
    means = []
    for matrix_list in matrix_lists:
        median_list = []
        mean_list = []
        for ix in range(len(matrix_list)):
            # n_samples_list.append(ix+1)

            # sum first how-many np.arrays
            summing_list = matrix_list[:ix + 1]
            sum_matrix = np.sum(summing_list, axis=0)
            frac_matrix = sum_matrix / (ix + 1)
            # exclude 0s
            frac_matrix = frac_matrix[frac_matrix != 0]
            # compute summary stats
            median = np.median(frac_matrix)
            mean = np.mean(frac_matrix)
            # breakpoint()

            # save them
            median_list.append(median)
            mean_list.append(mean)
        medians.append(median_list)
        means.append(mean_list)

    # breakpoint()
    df = pd.DataFrame.from_dict(
        {'alt_medians': medians[0], 'alt_means': means[0],
         'upwards_medians': medians[1], 'upwards_means': means[1],
         'dfs_medians': medians[2], 'dfs_means': means[2]}
    )

    return df


def make_edge_reuse_matrix_list_dfs(A, pred, num_solutions_extracted):
    """
    For a single graph & predicted probMatrix, plot Y-axis: edge reuse by X-axis: num samples

    Args:
        A: An adjacency matrix (np.array)
        pred: A probability distribution (list of lists, floats in each entry)
        num_solutions_extracted: An integer (e.g. 5) indicating the number of solutions to extract from pred

    Returns:
        matrices: list of adjacency matrices. Each adjacency matrix is a BF path (i.e. one solution).
    """
    # gather many solutions
    sol_counter = 0
    alt_matrices = []
    upwards_matrices = []
    dfs_matrices = []

    while sol_counter < num_solutions_extracted:
        sol_counter += 1
        alt_tree = get_parent_tree_upwards(pred)
        upwards_tree = single_sample_upwards(pred)
        dfs_matrix = dfs(A, deterministic=True)[0]

        # convert tree to adjacency matrix
        alt_matrix = parent_tree_to_adj_matrix(alt_tree)
        upwards_matrix = parent_tree_to_adj_matrix(upwards_tree)

        # save tree adj matrix
        alt_matrices.append(alt_matrix)
        upwards_matrices.append(upwards_matrix)
        dfs_matrices.append(dfs_matrix)
    # breakpoint()

    return [alt_matrices, upwards_matrices, dfs_matrices]


def DFS_plot(df):
    with plt.style.context(spstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots(ncols=1, sharey=True)
    plt.plot(df.index + 1, df.total_unique_seen, marker='o', linestyle='-')
    plt.axis((0, len(df), 0, len(df)))  # weird error, when I run in pycharm can't adjust axes, but works in terminal
    plt.title('num_unique by num_sampled')
    plt.xlabel('num_sampled')
    plt.ylabel('num_unique')
    plt.show()


# -----------------------------------------------------------------------------------------------------------------------
# BF GRAPH NUM UNIQUE by NUM SAMPLES
# - GOAL: find the num unique solutions extractable from distribution
# - WORKFLOW: Use `make_n_unique_by_n_extracted_df` to build df, then `plot_n_unique_by_n_extracted` to plot
# -----------------------------------------------------------------------------------------------------------------------

def make_n_unique_by_n_extracted_df_dfs(A, s, pred, num_solutions_extracted):
    """
    Makes DF with columns (e.g. 'total_uv_seen_upwards'), indicating the number of unique and valid solutions seen,
    when `df.index`-many solutions have been extracted.

    Args:
        A: an adjacency matrix
        s: starting node index (e.g. 5)
        pred: a predecessor array encoding a distribution of solutions (e.g. typical output of NN: [[],[],[]])
        num_solutions_extracted: a list of places where you want uniqueness evaluated (e.g. [5,25,100]

    Returns:
        df: pandas dataframe carrying info needed for plot

    """
    num_samples_drawn = 0
    #
    dfs_frequency_dict = dict()
    upwards_frequency_dict = dict()
    alt_frequency_dict = dict()
    alt_hashes = set()
    upwards_hashes = set()
    #
    upwards_valid = None
    alt_valid = None
    #
    dfs_trees_col = []
    alt_trees_col = []
    upwards_trees_col = []
    valid_alt = []
    valid_upwards = []
    #
    unique_dfs = []
    unique_alt = []
    unique_upwards = []
    times_found_alt = []
    times_found_upwards = []

    while num_samples_drawn < num_solutions_extracted:
        # take a sample, see if it's unique.
        tree1 = single_sample_upwards(pred)
        tree2 = get_parent_tree_upwards(pred)
        tree3 = adj_matrix_to_parent_tree(bellman_ford(A, s, deterministic=True)[0])
        #
        upwards_trees_col.append(tree1)
        alt_trees_col.append(tree2)
        dfs_trees_col.append(tree3)
        #

        # is it valid?
        valid_upwards.append(check_valid_BFpaths(A, s, tree1))
        valid_alt.append(check_valid_BFpaths(A, s, tree2))

        # how many times have we seen it before?
        hash_upwards = hash(tuple(tree1))
        hash_alt = hash(tuple(tree2))
        hash_dfs = hash(tuple(tree3))
        # breakpoint()

        # if not seen before, add 1. else add 0, so we can cumulatively sum df column
        unique_upwards.append(hash_upwards not in upwards_frequency_dict.keys())
        unique_alt.append(hash_alt not in alt_frequency_dict.keys())
        unique_dfs.append(hash_dfs not in dfs_frequency_dict.keys())

        # abcd?
        if hash_upwards not in upwards_frequency_dict.keys():
            upwards_frequency_dict[hash_upwards] = 1
        else:
            upwards_frequency_dict[hash_upwards] += 1

        if hash_alt not in alt_frequency_dict.keys():
            alt_frequency_dict[hash_alt] = 1
        else:
            alt_frequency_dict[hash_alt] += 1

        if hash_dfs not in dfs_frequency_dict.keys():
            dfs_frequency_dict[hash_dfs] = 1
        else:
            dfs_frequency_dict[hash_dfs] += 1

        # times_found_upwards.append()
        # times_found_alt.append()
        # TODO: SAVE THE DICTIONARY AT THE END, with tree specifics

        num_samples_drawn += 1
    # FIXME interval code nonsense

    df = pd.DataFrame.from_dict(
        {'upwards_sols': upwards_trees_col, 'unique_upwards': unique_upwards, 'valid_upwards': valid_upwards,
         'investment_bankers': alt_trees_col, 'unique_alt': unique_alt, 'valid_alt': valid_alt,
         'bellman_ford_sols': dfs_trees_col, 'unique_dfs': unique_dfs}
    )
    df['total_uv_seen_upwards'] = (df['unique_upwards'] & df['valid_upwards']).cumsum()
    df['total_uv_seen_alt'] = (df['unique_alt'] & df['valid_alt']).cumsum()
    df['total_uv_seen_dfs'] = df['unique_dfs'].cumsum()
    df.index += 1

    return df


def plot_n_unique_by_n_extracted_dfs(df, graphsize):
    """Plots a df produced by make_n_unique_by_n_extracted_df"""
    with plt.style.context(spstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots(ncols=1, sharey=True)
    df = pd.concat(df)
    # u & v
    total_uv_seen_upwards_mean = df['total_uv_seen_upwards'].groupby(df.index).mean()
    total_uv_seen_upwards_std = df['total_uv_seen_upwards'].groupby(df.index).std()

    total_uv_seen_alt_mean = df['total_uv_seen_alt'].groupby(df.index).mean()
    total_uv_seen_alt_std = df['total_uv_seen_alt'].groupby(df.index).std()

    total_uv_seen_dfs_mean = df['total_uv_seen_dfs'].groupby(df.index).mean()
    total_uv_seen_dfs_std = df['total_uv_seen_dfs'].groupby(df.index).std()

    plt.plot([i for i in range(1, len(total_uv_seen_upwards_mean)+1)], total_uv_seen_upwards_mean, marker='o', linestyle='-',
             color="blue", label="Upwards")
    plt.fill_between([i for i in range(1, len(total_uv_seen_upwards_mean)+1)],
                     total_uv_seen_upwards_mean - total_uv_seen_upwards_std,
                     total_uv_seen_upwards_mean + total_uv_seen_upwards_std, color="blue", alpha=0.15)

    plt.plot([i for i in range(1, len(total_uv_seen_upwards_mean)+1)], total_uv_seen_alt_mean, marker='x', linestyle='-',
             color="red", label="AltUpwards")
    plt.fill_between([i for i in range(1, len(total_uv_seen_upwards_mean)+1)],
                     total_uv_seen_alt_mean - total_uv_seen_alt_std,
                     total_uv_seen_alt_mean + total_uv_seen_alt_std, color="red", alpha=0.15)
    # plt.plot([i for i in range(len(total_uv_seen_upwards_mean))], total_uv_seen_alt_mean, marker='v', linestyle='-', color="green", label="DFS")
    plt.plot([i for i in range(1, len(total_uv_seen_upwards_mean)+1)], total_uv_seen_dfs_mean, marker='v', linestyle='-',
             color="green", label="DFS")
    plt.fill_between([i for i in range(1, len(total_uv_seen_dfs_mean)+1)],
                     total_uv_seen_dfs_mean - total_uv_seen_dfs_std,
                     total_uv_seen_dfs_mean + total_uv_seen_dfs_std, color="green", alpha=0.15)
    plt.legend(loc="upper left")
    # plt.axis((0, len(df), 0, len(df)))  # weird error, when I run in pycharm can't adjust axes, but works in terminal
    plt.title(f'Unique and valid solutions vs sampled solutions for n = {graphsize} (DFS)')
    plt.xlabel('Sampled solutions')
    plt.ylabel('Unique and valid solutions')
    plt.tight_layout()
    plt.savefig(f"plot_unique_by_extracted_{graphsize}_dfs.png")

def line_plot_dfs(df_list, graphsize):
    """
    Line Plots with Confidence Interval for this type of random graph (Confidence that mean lies within here) FIXME CIs are weird here
    Args:
        df_list: list of dfs, each df represents single graph, recording edge_reuse score over many samples
        graphsize: helps name the file

    Returns: None, just plots
    """
    # Need, at each df.index, a +/- for CI
    num_graphs = len(df_list)   # sample size

    with plt.style.context(spstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots(ncols=1, sharey=True)
    df = pd.concat(df_list)
    #breakpoint()
    # u & v
    mean_edge_reuse_upwards_mean = df['upwards_means'].groupby(df.index).mean()
    mean_edge_reuse_upwards_std = df['upwards_means'].groupby(df.index).std()

    mean_edge_reuse_alt_mean = df['alt_means'].groupby(df.index).mean()
    mean_edge_reuse_alt_std = df['alt_means'].groupby(df.index).std()

    mean_edge_reuse_dfs_mean = df['dfs_means'].groupby(df.index).mean()
    mean_edge_reuse_dfs_std = df['dfs_means'].groupby(df.index).std()

    means = [mean_edge_reuse_upwards_mean, mean_edge_reuse_alt_mean, mean_edge_reuse_dfs_mean]
    std = [mean_edge_reuse_upwards_std, mean_edge_reuse_alt_std, mean_edge_reuse_dfs_std]


    plt.plot([i for i in range(1, len(mean_edge_reuse_upwards_mean)+1)], mean_edge_reuse_upwards_mean, marker='o', linestyle='-',color="blue", label="Upwards")
    plt.fill_between([i for i in range(1, len(mean_edge_reuse_upwards_mean)+1)], mean_edge_reuse_upwards_mean - mean_edge_reuse_upwards_std,mean_edge_reuse_upwards_mean + mean_edge_reuse_upwards_std, color="blue", alpha=0.15)

    plt.plot([i for i in range(1, len(mean_edge_reuse_upwards_mean)+1)], mean_edge_reuse_alt_mean, marker='x', linestyle='-', color="red", label="AltUpwards")
    plt.fill_between([i for i in range(1, len(mean_edge_reuse_upwards_mean)+1)], mean_edge_reuse_alt_mean - mean_edge_reuse_alt_std,mean_edge_reuse_alt_mean + mean_edge_reuse_alt_std, color="red", alpha=0.15)
    #plt.plot([i for i in range(len(total_uv_seen_upwards_mean))], total_uv_seen_alt_mean, marker='v', linestyle='-', color="green", label="Bellman-Ford")
    plt.plot([i for i in range(1, len(mean_edge_reuse_upwards_mean)+1)], mean_edge_reuse_dfs_mean, marker='v', linestyle='-', color="green", label="DFS")
    plt.fill_between([i for i in range(1, len(mean_edge_reuse_upwards_mean)+1)],mean_edge_reuse_dfs_mean - mean_edge_reuse_dfs_std,mean_edge_reuse_dfs_mean + mean_edge_reuse_dfs_std, color="green", alpha=0.15)
    plt.legend(loc="lower left")
    plt.axis((0, len(mean_edge_reuse_upwards_mean), 0, 1))
    # plt.plot(df.n_samples, df.medians, marker='o', linestyle='-')
    # plt.axis((0, len(df), 0, 1))  # weird error, when I run in pycharm can't adjust axes, but works in terminal
    plt.title(f'Mean average edge reuse by sampling method for n = {graphsize} (DFS)')
    plt.ylabel('Mean average edge reuse')
    plt.xlabel('Number of solutions extracted')
    plt.tight_layout()
    plt.savefig(f"edge_reuse_lineplot{graphsize}_dfs.png")
    plt.close()


def average_dataframes(df_list):
    #breakpoint()
    combined_df = pd.concat(df_list)
    # u & v
    mean_unique_and_valid = combined_df['unique_and_valid'].groupby(combined_df.index).mean()
    median_unique_and_valid = combined_df['unique_and_valid'].groupby(combined_df.index).mean()
    max_uv = combined_df['unique_and_valid'].groupby(combined_df.index).max()
    min_uv = combined_df['unique_and_valid'].groupby(combined_df.index).min()
    # just unique
    mean_unique = combined_df['total_unique_seen'].groupby(combined_df.index).mean()
    median_unique = combined_df['total_unique_seen'].groupby(combined_df.index).mean()
    max_u = combined_df['total_unique_seen'].groupby(combined_df.index).max()
    min_u = combined_df['total_unique_seen'].groupby(combined_df.index).min()
    # just valid
    mean_valid = combined_df['total_valid_seen'].groupby(combined_df.index).mean()
    median_valid = combined_df['total_valid_seen'].groupby(combined_df.index).mean()
    max_v = combined_df['total_valid_seen'].groupby(combined_df.index).max()
    min_v = combined_df['total_valid_seen'].groupby(combined_df.index).min()
    # summarize

    df = pd.DataFrame.from_dict({'mean_unique_and_valid': mean_unique_and_valid,
                                 'median_unique_and_valid': median_unique_and_valid,
                                 'max_uv': max_uv,
                                 'min_uv': min_uv,
                                 'mean_unique': mean_unique,
                                 'median_unique': median_unique,
                                 'max_u': max_u,
                                 'min_u': min_u,
                                 'mean_valid': mean_valid,
                                 'median_valid': median_valid,
                                 'max_v': max_v,
                                 'min_v': min_v
                                 })
    df['samples_seen'] = df.index+1
    return df

def DFS_plot(df):
    with plt.style.context(spstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots(ncols=1, sharey=True)
    plt.plot(df.index + 1, df.total_unique_seen, marker='o', linestyle='-')
    plt.axis((0, len(df), 0, len(df)))  # weird error, when I run in pycharm can't adjust axes, but works in terminal
    plt.title('num_unique by num_sampled')
    plt.xlabel('num_sampled')
    plt.ylabel('num_unique')
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# BF GRAPH NUM UNIQUE by NUM SAMPLES
# - GOAL: find the num unique solutions extractable from distribution
# - WORKFLOW: Use `make_n_unique_by_n_extracted_df` to build df, then `plot_n_unique_by_n_extracted` to plot
#-----------------------------------------------------------------------------------------------------------------------

def make_n_unique_by_n_extracted_df(A, s, pred, num_solutions_extracted):
    """
    Makes DF with columns (e.g. 'total_uv_seen_beam'), indicating the number of unique and valid solutions seen,
    when `df.index`-many solutions have been extracted.

    Args:
        A: an adjacency matrix
        s: starting node index (e.g. 5)
        pred: a predecessor array encoding a distribution of solutions (e.g. typical output of NN: [[],[],[]])
        num_solutions_extracted: a list of places where you want uniqueness evaluated (e.g. [5,25,100]

    Returns:
        df: pandas dataframe carrying info needed for plot

    """
    num_samples_drawn = 0
    #
    bf_frequency_dict = dict()
    beam_frequency_dict = dict()
    greedy_frequency_dict = dict()
    greedy_hashes = set()
    beam_hashes = set()
    #
    beam_valid = None
    greedy_valid = None
    #
    bf_trees_col = []
    greedy_trees_col = []
    beam_trees_col = []
    valid_greedy = []
    valid_beam = []
    #
    unique_bf = []
    unique_greedy = []
    unique_beam = []
    times_found_greedy = []
    times_found_beam = []

    while num_samples_drawn < num_solutions_extracted:
        # take a sample, see if it's unique.
        tree1 = BF_beamsearch(A, s, pred)
        tree2 = BF_greedysearch(A, s, pred)
        tree3 = adj_matrix_to_parent_tree(bellman_ford(A,s,deterministic=True)[0])
        #
        beam_trees_col.append(tree1)
        greedy_trees_col.append(tree2)
        bf_trees_col.append(tree3)
        #

        # is it valid?
        valid_beam.append(check_valid_BFpaths(A, s, tree1))
        valid_greedy.append(check_valid_BFpaths(A, s, tree2))


        # how many times have we seen it before?
        hash_beam = hash(tuple(tree1))
        hash_greedy = hash(tuple(tree2))
        hash_bf = hash(tuple(tree3))
        #breakpoint()

        # if not seen before, add 1. else add 0, so we can cumulatively sum df column
        unique_beam.append(hash_beam not in beam_frequency_dict.keys())
        unique_greedy.append(hash_greedy not in greedy_frequency_dict.keys())
        unique_bf.append(hash_bf not in bf_frequency_dict.keys())

        # abcd?
        if hash_beam not in beam_frequency_dict.keys():
            beam_frequency_dict[hash_beam] = 1
        else:
            beam_frequency_dict[hash_beam] += 1

        if hash_greedy not in greedy_frequency_dict.keys():
            greedy_frequency_dict[hash_greedy] = 1
        else:
            greedy_frequency_dict[hash_greedy] += 1

        if hash_bf not in bf_frequency_dict.keys():
            bf_frequency_dict[hash_bf] = 1
        else:
            bf_frequency_dict[hash_bf] += 1

        #times_found_beam.append()
        #times_found_greedy.append()
        # TODO: SAVE THE DICTIONARY AT THE END, with tree specifics

        num_samples_drawn += 1
    # FIXME interval code nonsense

    df = pd.DataFrame.from_dict(
            {'beam_sols': beam_trees_col, 'unique_beam':unique_beam, 'valid_beam':valid_beam,
             'investment_bankers': greedy_trees_col, 'unique_greedy': unique_greedy, 'valid_greedy': valid_greedy,
             'bellman_ford_sols':bf_trees_col, 'unique_bf':unique_bf}
    )
    df['total_uv_seen_beam'] = (df['unique_beam'] & df['valid_beam']).cumsum()
    df['total_uv_seen_greedy'] = (df['unique_greedy'] & df['valid_greedy']).cumsum()
    df['total_unique_seen_bf'] = df['unique_bf'].cumsum()
    df.index += 1

    return df


def plot_n_unique_by_n_extracted(df, graphsize):
    """Plots a df produced by make_n_unique_by_n_extracted_df"""
    with plt.style.context(spstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots(ncols=1, sharey=True)
    df = pd.concat(df)
    # u & v
    total_uv_seen_beam_mean = df['total_uv_seen_beam'].groupby(df.index).mean()
    total_uv_seen_beam_std = df['total_uv_seen_beam'].groupby(df.index).std()

    total_uv_seen_greedy_mean = df['total_uv_seen_greedy'].groupby(df.index).mean()
    total_uv_seen_greedy_std = df['total_uv_seen_greedy'].groupby(df.index).std()

    total_uv_seen_bf_mean = df['total_unique_seen_bf'].groupby(df.index).mean()
    total_uv_seen_bf_std = df['total_unique_seen_bf'].groupby(df.index).std()

    plt.plot([i for i in range(len(total_uv_seen_beam_mean))], total_uv_seen_beam_mean, marker='o', linestyle='-', color = "blue", label = "Beamsearch")
    plt.fill_between([i for i in range(len(total_uv_seen_beam_mean))], total_uv_seen_beam_mean-total_uv_seen_beam_std, total_uv_seen_beam_mean+total_uv_seen_beam_std, color = "blue", alpha = 0.15)

    plt.plot([i for i in range(len(total_uv_seen_beam_mean))], total_uv_seen_greedy_mean, marker='x', linestyle='-', color="red", label = "Greedy")
    plt.fill_between([i for i in range(len(total_uv_seen_beam_mean))], total_uv_seen_greedy_mean - total_uv_seen_greedy_std,
                     total_uv_seen_greedy_mean + total_uv_seen_greedy_std, color="red", alpha=0.15)
    #plt.plot([i for i in range(len(total_uv_seen_beam_mean))], total_uv_seen_greedy_mean, marker='v', linestyle='-', color="green", label="Bellman-Ford")
    plt.plot([i for i in range(len(total_uv_seen_beam_mean))], total_uv_seen_bf_mean, marker='v', linestyle='-',
             color="green", label="Bellman-Ford")
    plt.fill_between([i for i in range(len(total_uv_seen_bf_mean))],
                     total_uv_seen_bf_mean - total_uv_seen_bf_std,
                     total_uv_seen_bf_mean + total_uv_seen_bf_std, color="red", alpha=0.15)
    plt.legend(loc = "upper left")
    #plt.axis((0, len(df), 0, len(df)))  # weird error, when I run in pycharm can't adjust axes, but works in terminal
    plt.title(f'Unique and valid solutions vs sampled solutions for n = {graphsize} (BF)')
    plt.xlabel('Sampled solutions')
    plt.ylabel('Unique and valid solutions')
    plt.tight_layout()
    plt.savefig(f"plot_unique_by_extracted_{graphsize}.png")


# ----------------------------------------------------------------------------------------------------------------------
# GRAPH EDGE REUSE BY NUM SAMPLES
# - GOAL: test similarity among solutions
# - WORKFLOW: `make_edge_reuse_matrix_list` to `postprocess_edge_reuse_matrix_list` to
#       `plot_edge_reuse_matrix_list_mean` for BAR CHART
# ----------------------------------------------------------------------------------------------------------------------
def make_edge_reuse_matrix_list(A, s, pred, num_solutions_extracted):
    """
    For a single graph & predicted probMatrix, plot Y-axis: edge reuse by X-axis: num samples

    Args:
        A: An adjacency matrix (np.array)
        s: An integer (e.g. 4) representing the index of starting node
        pred: A probability distribution (list of lists, floats in each entry)
        num_solutions_extracted: An integer (e.g. 5) indicating the number of solutions to extract from pred

    Returns:
        matrices: list of adjacency matrices. Each adjacency matrix is a BF path (i.e. one solution).
    """
    # gather many solutions
    sol_counter = 0
    greedy_matrices = []
    beam_matrices = []
    bf_matrices = []

    while sol_counter < num_solutions_extracted:
        sol_counter += 1
        greedy_tree = BF_greedysearch(A, s, pred)
        beam_tree = BF_beamsearch(A, s, pred)
        bf_matrix = bellman_ford(A, s, deterministic=True)[0]

        # convert tree to adjacency matrix
        greedy_matrix = parent_tree_to_adj_matrix(greedy_tree)
        beam_matrix = parent_tree_to_adj_matrix(beam_tree)

        # save tree adj matrix
        greedy_matrices.append(greedy_matrix)
        beam_matrices.append(beam_matrix)
        bf_matrices.append(bf_matrix)
    #breakpoint()

    return [greedy_matrices, beam_matrices, bf_matrices]


def postprocess_edge_reuse_matrix_list(matrix_lists):
    """
    Convert many solutions (matrix_lists) from a single graph & predicted probMatrix, into a df of means by n_sols
    Args:
        matrix_lists: a list of lists [[greedy_list], [beam_list], [bf_list]]
            each inner list (e.g. [greedy_list]) contains adjacency matrices. Each adjacency matrix is a BF path.

    Returns:
        df: pandas Dataframe with greedy, beam, and bf means and medians
    """
    # this is all for a single graph, where each matrix represents a subset of edges... perhaps in graph... extracted from parent tree
    # at each interval, sum adjacency matrices, calculate frequency, report score
    # TODO: filter by validity?

    #breakpoint()
    #n_samples_list = []
    medians = []
    means = []
    for matrix_list in matrix_lists:
        median_list = []
        mean_list = []
        for ix in range(len(matrix_list)):
            #n_samples_list.append(ix+1)

            # sum first how-many np.arrays
            summing_list = matrix_list[:ix+1]
            sum_matrix = np.sum(summing_list, axis=0)
            frac_matrix = sum_matrix/(ix+1)
            # exclude 0s
            frac_matrix = frac_matrix[frac_matrix != 0]
            # compute summary stats
            median = np.median(frac_matrix)
            mean = np.mean(frac_matrix)
            #breakpoint()

            # save them
            median_list.append(median)
            mean_list.append(mean)
        medians.append(median_list)
        means.append(mean_list)

    #breakpoint()
    df = pd.DataFrame.from_dict(
        {'greedy_medians': medians[0], 'greedy_means': means[0],
         'beam_medians': medians[1], 'beam_means': means[1],
         'bf_medians': medians[2], 'bf_means': means[2]}
        )

    return df


def plot_edge_reuse_matrix_list_mean(df, graphsize):
    """FIXME: .iloc[-1] is taking only the last number? after the max number of solutions."""
    with plt.style.context(spstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots(ncols=1, sharey=True)
    df = pd.concat(df)
    # u & v
    mean_edge_reuse_beam_mean = df['beam_means'].groupby(df.index).mean().iloc[-1]
    mean_edge_reuse_beam_std = df['beam_means'].groupby(df.index).std().iloc[-1]

    mean_edge_reuse_greedy_mean = df['greedy_means'].groupby(df.index).mean().iloc[-1]
    mean_edge_reuse_greedy_std = df['greedy_means'].groupby(df.index).std().iloc[-1]

    mean_edge_reuse_bf_mean = df['bf_means'].groupby(df.index).mean().iloc[-1]
    mean_edge_reuse_bf_std = df['bf_means'].groupby(df.index).std().iloc[-1]

    means = [mean_edge_reuse_beam_mean,mean_edge_reuse_greedy_mean,mean_edge_reuse_bf_mean]
    std = [mean_edge_reuse_beam_std, mean_edge_reuse_greedy_std, mean_edge_reuse_bf_std]

    plt.bar(np.arange(len(means)), means, 0.4)
    plt.errorbar(np.arange(len(means)), means, yerr=std,color = "black",capsize = 5, ls = "None")
    plt.xticks(np.arange(len(means)),["Beam","Greedy", "Bellman-Ford"])

    #plt.plot([i for i in range(len(mean_edge_reuse_beam_mean))], mean_edge_reuse_beam_mean, marker='o', linestyle='-',color="blue", label="Beamsearch")
    #plt.fill_between([i for i in range(len(mean_edge_reuse_beam_mean))], mean_edge_reuse_beam_mean - mean_edge_reuse_beam_std,mean_edge_reuse_beam_mean + mean_edge_reuse_beam_std, color="blue", alpha=0.15)

    #plt.plot([i for i in range(len(mean_edge_reuse_beam_mean))], mean_edge_reuse_greedy_mean, marker='x', linestyle='-', color="red", label="Greedy")
   # plt.fill_between([i for i in range(len(mean_edge_reuse_beam_mean))],mean_edge_reuse_greedy_mean - mean_edge_reuse_greedy_std,mean_edge_reuse_greedy_mean + mean_edge_reuse_greedy_std, color="red", alpha=0.15)
    # plt.plot([i for i in range(len(total_uv_seen_beam_mean))], total_uv_seen_greedy_mean, marker='v', linestyle='-', color="green", label="Bellman-Ford")
    #plt.plot([i for i in range(len(mean_edge_reuse_beam_mean))], mean_edge_reuse_bf_mean, marker='v', linestyle='-', color="green", label="Bellman-Ford")
    #plt.fill_between([i for i in range(len(mean_edge_reuse_beam_mean))],mean_edge_reuse_bf_mean - mean_edge_reuse_bf_std,mean_edge_reuse_bf_mean + mean_edge_reuse_bf_std, color="red", alpha=0.15)
    #plt.legend(loc="upper left")
    #plt.plot(df.n_samples, df.medians, marker='o', linestyle='-')
    #plt.axis((0, len(df), 0, 1))  # weird error, when I run in pycharm can't adjust axes, but works in terminal
    plt.title(f'Mean average edge reuse for n = {graphsize} (BF)')
    plt.ylabel('Mean average edge reuse')
    plt.tight_layout()
    plt.savefig("edge_reuse_mean_"+str(graphsize)+".png")
    plt.close()


def line_plot(df_list, graphsize):
    """
    Line Plots with Confidence Interval for this type of random graph (Confidence that mean lies within here) FIXME CIs are weird here
    Args:
        df_list: list of dfs, each df represents single graph, recording edge_reuse score over many samples
        graphsize: helps name the file

    Returns: None, just plots
    """
    # Need, at each df.index, a +/- for CI
    num_graphs = len(df_list)   # sample size

    with plt.style.context(spstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots(ncols=1, sharey=True)
    df = pd.concat(df_list)
    #breakpoint()
    # u & v
    mean_edge_reuse_beam_mean = df['beam_means'].groupby(df.index).mean()
    mean_edge_reuse_beam_std = df['beam_means'].groupby(df.index).std()

    mean_edge_reuse_greedy_mean = df['greedy_means'].groupby(df.index).mean()
    mean_edge_reuse_greedy_std = df['greedy_means'].groupby(df.index).std()

    mean_edge_reuse_bf_mean = df['bf_means'].groupby(df.index).mean()
    mean_edge_reuse_bf_std = df['bf_means'].groupby(df.index).std()

    means = [mean_edge_reuse_beam_mean, mean_edge_reuse_greedy_mean, mean_edge_reuse_bf_mean]
    std = [mean_edge_reuse_beam_std, mean_edge_reuse_greedy_std, mean_edge_reuse_bf_std]


    plt.plot([i for i in range(1, len(mean_edge_reuse_beam_mean)+1)], mean_edge_reuse_beam_mean, marker='o', linestyle='-',color="blue", label="Beamsearch")
    plt.fill_between([i for i in range(1, len(mean_edge_reuse_beam_mean)+1)], mean_edge_reuse_beam_mean - mean_edge_reuse_beam_std,mean_edge_reuse_beam_mean + mean_edge_reuse_beam_std, color="blue", alpha=0.15)

    plt.plot([i for i in range(1, len(mean_edge_reuse_beam_mean)+1)], mean_edge_reuse_greedy_mean, marker='x', linestyle='-', color="red", label="Greedy")
    plt.fill_between([i for i in range(1, len(mean_edge_reuse_beam_mean)+1)], mean_edge_reuse_greedy_mean - mean_edge_reuse_greedy_std,mean_edge_reuse_greedy_mean + mean_edge_reuse_greedy_std, color="red", alpha=0.15)
    #plt.plot([i for i in range(len(total_uv_seen_beam_mean))], total_uv_seen_greedy_mean, marker='v', linestyle='-', color="green", label="Bellman-Ford")
    plt.plot([i for i in range(1, len(mean_edge_reuse_beam_mean)+1)], mean_edge_reuse_bf_mean, marker='v', linestyle='-', color="green", label="Bellman-Ford")
    plt.fill_between([i for i in range(1, len(mean_edge_reuse_beam_mean)+1)],mean_edge_reuse_bf_mean - mean_edge_reuse_bf_std,mean_edge_reuse_bf_mean + mean_edge_reuse_bf_std, color="green", alpha=0.15)
    plt.legend(loc="lower left")
    plt.axis((0, len(mean_edge_reuse_beam_mean), 0,1))
    # plt.plot(df.n_samples, df.medians, marker='o', linestyle='-')
    # plt.axis((0, len(df), 0, 1))  # weird error, when I run in pycharm can't adjust axes, but works in terminal
    plt.title(f'Mean average edge reuse by sampling method for n = {graphsize} (BF)')
    plt.ylabel('Mean average edge reuse')
    plt.xlabel('Number of solutions extracted')
    plt.tight_layout()
    plt.savefig("edge_reuse_lineplot" + str(graphsize) + ".png")
    plt.close()



# TODO: graph edge reuse & sort it by algorithm training?
# TODO: measure total number of distinct edges or some other metric to see solution diversity?
# TODO: change graph style or graph size...
# TODO: compare to Empirical Distribution (by plugging-in outs in collect_and_eval)



if __name__ == '__main__':

    print('testing _src/validate_distributions.py')

    test_A = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])

    test_s = 1

    test_pred = np.array([
        [0.1, 0.3, 0.6],
        [0.2, 0.8, 0.0],
        [0.4, 0.5, 0.1]
    ])

    test_intervals = 50

    '''
    df1 = make_n_unique_by_n_extracted_df(A=test_A, s=test_s, pred=test_pred, num_solutions_extracted=test_intervals)
    plot_n_unique_by_n_extracted(df1)

    ms = make_edge_reuse_matrix_list(A=test_A, s=test_s, pred=test_pred, num_solutions_extracted=test_intervals)
    df = postprocess_edge_reuse_matrix_list(ms)
    plot_edge_reuse_matrix_list(df)
    '''
    df_list = []
    for i in range(100):
        df, A, pM = DFS_graph1_df(A=test_A, pred=test_pred, num_solutions_extracted=test_intervals)
        df_list.append(df)
    df = average_dataframes(df_list)


    plt.plot(df.samples_seen, df.median_unique, marker='o', linestyle='-')
