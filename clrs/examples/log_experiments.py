import os
import time
from datetime import datetime

import numpy as np
import clrs     # for clrs.evaluate
import jax
import pandas as pd

from clrs._src.multi_sol.sampling import dfs as dfs_sampling
from clrs._src.multi_sol.sampling import bellman_ford as bf_sampling
from clrs._src.multi_sol.algorithms.bfs.plugin import evaluate_bfs_multisol_batch
from clrs._src.multi_sol.algorithms.dfs.plugin import evaluate_dfs_multisol_batch
from clrs._src.multi_sol.algorithms.bellman_ford.plugin import evaluate_bf_multisol_batch
from clrs._src import dfs_uniqueness_check
from clrs._src.algorithms import check_graphs, dfs_verification_tester
from clrs._src.bf_uniqueness_check import check_uniqueness_bf
from clrs._src.multi_sol.data.distribution_generation import (
    build_bf_validation_payload,
    build_dfs_validation_payload,
    generate_validation_dataframes,
)
#from clrs.examples.run import _concat, unpack  # circular import error!

from clrs._src.validate_distributions import (
    line_plot,
    line_plot_dfs,
    plot_edge_reuse_matrix_list_mean,
    plot_edge_reuse_matrix_list_mean_dfs,
    plot_n_unique_by_n_extracted,
    plot_n_unique_by_n_extracted_dfs,
)

###############################################################
# Helpers
###############################################################
def save_results(result_dict, filename):
    """Save result_dict as a timestamped CSV in the results/ directory."""
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join('results', f'{filename}_{timestamp}.csv')
    pd.DataFrame.from_dict(result_dict).to_csv(save_path, encoding='utf-8', index=False)
    print(f'Saved results to {save_path}')


###############################################################
# Methods needed, copy-pasted from run.py :(
###############################################################
def _concat(dps, axis):
  return jax.tree_util.tree_map(lambda *x: np.concatenate(x, axis), *dps)

def unpack(v):
  try:
    return v.item()  # DeviceArray
  except (AttributeError, ValueError):
    return v


###############################################################
# BF pipeline
###############################################################

def BF_collect_and_eval(sampler, predict_fn, sample_count, rng_key, extras, filename='bf_accuracy', vd_flag=True, NSE = 100):
    """Collect batches of output and hint preds and evaluate them."""
    if not vd_flag:
        return evaluate_bf_multisol_batch(
            sampler=sampler,
            predict_fn=predict_fn,
            sample_count=sample_count,
            rng_key=rng_key,
            extras=extras,
            save_results_fn=save_results,
            filename=filename,
        )
    processed_samples = 0
    preds = []
    outputs = []
    As = []
    source_nodes = []
    while processed_samples < sample_count:
        feedback = next(sampler)
        batch_size = feedback.outputs[0].data.shape[0]
        outputs.append(feedback.outputs)
        new_rng_key, rng_key = jax.random.split(rng_key)
        cur_preds, _ = predict_fn(new_rng_key, feedback.features)
        preds.append(cur_preds)
        processed_samples += batch_size
        As.append(feedback[0][0][2].data)
        source_nodes.append(
            np.argmax(feedback[0][0][1].data, axis=1))  # todo checkme. feedback[0][0][1] is 's', .data gives array
    outputs = _concat(outputs, axis=0)
    As = _concat(As, axis=0)  # concatenate batches
    source_nodes = _concat(source_nodes, axis=0)
    # breakpoint()
    preds = _concat(preds, axis=0)
    out = clrs.evaluate(outputs, preds)

    #breakpoint()
    if vd_flag:
        print('log_exp.py, vd_flag working')
        print('log_exp.py, BF_collect_and_eval, only testing on 10 of the graphs')
        payload = build_bf_validation_payload(
            adjacency=As,
            source_nodes=source_nodes,
            outputs=outputs,
            preds=preds,
        )
        dataframes, df = generate_validation_dataframes(
            payload=payload,
            nse=NSE,
            mode='BF',
        )
        plot_n_unique_by_n_extracted(dataframes, len(As[0]))

        #plot_edge_reuse_matrix_list_median(df, len(As[0]))
        #breakpoint()
        plot_edge_reuse_matrix_list_mean(df, payload.graph_size)
        line_plot(df, payload.graph_size)
        #breakpoint()

    ########
    # RANDOM #
    ########
    model_sample_random = dfs_sampling.sample_random_list([preds])
    true_sample_random = dfs_sampling.sample_random_list(outputs)

    model_random_truthmask = [check_graphs.check_valid_BFpaths(As[i], source_nodes[i], model_sample_random[i]) for i in range(len(model_sample_random))]
    correctness_model_random = sum(model_random_truthmask) / len(model_random_truthmask)

    true_random_truthmask = [check_graphs.check_valid_BFpaths(As[i], source_nodes[i], true_sample_random[i]) for i in range(len(true_sample_random))]
    correctness_true_random = sum(true_random_truthmask) / len(true_random_truthmask)

    #breakpoint()
    ########
    # BEAM #
    ########
    model_sample_beam = bf_sampling.sample_beamsearch(As, source_nodes, [preds])
    true_sample_beam = bf_sampling.sample_beamsearch(As, source_nodes, outputs)

    model_beam_truthmask = [check_graphs.check_valid_BFpaths(As[i], source_nodes[i], model_sample_beam[i]) for i in range(len(model_sample_beam))]
    correctness_model_beam = sum(model_beam_truthmask) / len(model_beam_truthmask)

    true_beam_truthmask = [check_graphs.check_valid_BFpaths(As[i], source_nodes[i], true_sample_beam[i]) for i in range(len(model_sample_random))]
    correctness_true_beam = sum(true_beam_truthmask) / len(true_beam_truthmask)

    model_beam_uniques, model_beam_valids_uniques, model_beam_valids = check_uniqueness_bf(As=As, source_nodes=source_nodes,
                                                                                           probMatrices = [preds], method="beam")
    true_beam_uniques, true_beam_valids_uniques, true_beam_valids = check_uniqueness_bf(As=As, source_nodes=source_nodes,
                                                                                           probMatrices = outputs, method="beam")

    #breakpoint()
    ########
    # Argmax #
    ########
    model_sample_argmax = dfs_sampling.sample_argmax([preds]) # doesn't use A or s
    true_sample_argmax = dfs_sampling.sample_argmax(outputs)

    model_argmax_truthmask = [check_graphs.check_valid_BFpaths(As[i], source_nodes[i], model_sample_argmax[i]) for i in
                            range(len(model_sample_argmax))]
    correctness_model_argmax = sum(model_argmax_truthmask) / len(model_argmax_truthmask)

    true_argmax_truthmask = [check_graphs.check_valid_BFpaths(As[i], source_nodes[i], true_sample_argmax[i]) for i in
                           range(len(model_sample_random))]
    correctness_true_argmax = sum(true_argmax_truthmask) / len(true_argmax_truthmask)

    #breakpoint()
    ########
    # greedy beam #
    ########
    model_sample_greedy = bf_sampling.sample_greedysearch(As, source_nodes, [preds])
    true_sample_greedy = bf_sampling.sample_greedysearch(As, source_nodes, outputs)

    model_greedy_truthmask = [check_graphs.check_valid_BFpaths(As[i], source_nodes[i], model_sample_greedy[i]) for i in
                            range(len(model_sample_greedy))]
    correctness_model_greedy = sum(model_greedy_truthmask) / len(model_greedy_truthmask)

    true_greedy_truthmask = [check_graphs.check_valid_BFpaths(As[i], source_nodes[i], true_sample_greedy[i]) for i in
                           range(len(model_sample_random))]
    correctness_true_greedy = sum(true_greedy_truthmask) / len(true_greedy_truthmask)

    model_greedy_uniques, model_greedy_valids_uniques, model_greedy_valids = check_uniqueness_bf(As = As, source_nodes = source_nodes,
        probMatrices=[preds], method = "greedy")
    true_greedy_uniques, true_greedy_valids_uniques, true_greedy_valids = check_uniqueness_bf(As = As, source_nodes = source_nodes,
        probMatrices=outputs, method = "greedy")


    ### LOGGING ###
    As = [i.flatten() for i in As]
    result_dict = {"As": As,
                   #
                   "Argmax_Model_Trees": model_sample_argmax,
                   "Argmax_True_Trees": true_sample_argmax,
                   #
                   "Argmax_Model_Mask": model_argmax_truthmask,
                   "Argmax_True_Mask": true_argmax_truthmask,
                   #
                   "Argmax_Model_Accuracy": correctness_model_argmax,
                   "Argmax_True_Accuracy": correctness_true_argmax,
                   #
                   ###
                   #
                   "Random_Model_Trees": model_sample_random,
                   "Random_True_Trees": true_sample_random,
                   #
                   "Random_Model_Mask": model_random_truthmask,
                   "Random_True_Mask": true_random_truthmask,
                   #
                   "Random_Model_Accuracy": correctness_model_random,
                   "Random_True_Accuracy": correctness_true_random,
                   #
                   ###
                   #
                   "Beam_Model_Trees": model_sample_beam,
                   "Beam_True_Trees": true_sample_beam,
                   #
                   "Beam_Model_Mask": model_beam_truthmask,
                   "Beam_True_Mask": true_beam_truthmask,
                   #
                   "Beam_Model_Accuracy": correctness_model_beam,
                   "Beam_True_Accuracy": correctness_true_beam,

                   "Beam_Model_Uniques": model_beam_uniques,
                   "Beam_Model_Valids_Uniques": model_beam_valids_uniques,
                   "Beam_Model_Valids": model_beam_valids,

                   "Beam_True_Uniques": true_beam_uniques,
                   "Beam_True_Valids_Uniques": true_beam_valids_uniques,
                   "Beam_True_Valids": true_beam_valids,
                   #
                   ###
                   #
                   "Greedy_Model_Trees": model_sample_greedy,
                   "Greedy_True_Trees": true_sample_greedy,
                   #
                   "Greedy_Model_Mask": model_greedy_truthmask,
                   "Greedy_True_Mask": true_greedy_truthmask,
                   #
                   "Greedy_Model_Accuracy": correctness_model_greedy,
                   "Greedy_True_Accuracy": correctness_true_greedy,

                   "Greedy_Model_Uniques":model_greedy_uniques,
                   "Greedy_Model_Valids_Uniques": model_greedy_valids_uniques,
                   "Greedy_Model_Valids": model_greedy_valids,

                   "Greedy_True_Uniques": true_greedy_uniques,
                   "Greedy_True_Valids_Uniques": true_greedy_valids_uniques,
                   "Greedy_True_Valids": true_greedy_valids
                   #
                   }
    save_results(result_dict, f"{filename}_BF")

    if extras:
        out.update(extras)
    return {k: unpack(v) for k, v in out.items()}


###############################################################
# BFS Multi-Solution
###############################################################

def BFS_multi_collect_and_eval(sampler, predict_fn, sample_count, rng_key, extras, 
                                filename='bfs_accuracy', vd_flag=False, NSE=100):
    """Collect batches of output and evaluate BFS multi-solution."""
    del vd_flag, NSE  # kept for CLI compatibility
    return evaluate_bfs_multisol_batch(
        sampler=sampler,
        predict_fn=predict_fn,
        sample_count=sample_count,
        rng_key=rng_key,
        extras=extras,
        save_results_fn=save_results,
        filename=filename,
    )


###############################################################
# DFS
###############################################################

def DFS_collect_and_eval(sampler, predict_fn, sample_count, rng_key, extras, filename = 'dfs_accuracy', vd_flag=False, NSE = 100):
    """Collect batch of output preds and evaluate them."""
    if not vd_flag:
        return evaluate_dfs_multisol_batch(
            sampler=sampler,
            predict_fn=predict_fn,
            sample_count=sample_count,
            rng_key=rng_key,
            extras=extras,
            save_results_fn=save_results,
            filename=filename,
        )
    processed_samples = 0
    preds = []
    outputs = []
    As = []
    while processed_samples < sample_count:
        feedback = next(sampler)
        batch_size = feedback.outputs[0].data.shape[0]
        outputs.append(feedback.outputs)
        new_rng_key, rng_key = jax.random.split(rng_key)
        cur_preds, _ = predict_fn(new_rng_key, feedback.features)
        preds.append(cur_preds)
        processed_samples += batch_size
        As.append(feedback[0][0][1].data)
    outputs = _concat(outputs, axis=0)
    As = _concat(As, axis=0) # concatenate batches
    #breakpoint()

    if vd_flag:
        print('log_exp.py, vd_flag working')
        print('truncating DFS_collect_and_eval to 10 graphs')
        payload = build_dfs_validation_payload(
            adjacency=As,
            outputs=outputs,
            preds=preds,
        )
        dataframes, df = generate_validation_dataframes(
            payload=payload,
            nse=NSE,
            mode='DFS',
        )
        #breakpoint()
        plot_n_unique_by_n_extracted_dfs(dataframes, len(As[0]))

        # plot_edge_reuse_matrix_list_median(df, len(As[0]))
        # breakpoint()
        plot_edge_reuse_matrix_list_mean_dfs(df, payload.graph_size)
        line_plot_dfs(df, payload.graph_size)

    ### We need preds and A. We want to
    # 1. Sample from preds a candidate tree
    # 2. run check_graphs on candidate tree (using A as groundtruth)
    # 3. Collect validity result into a dataframe.

    t1 = time.time()
    ##### RANDOM
    model_sample_random = dfs_sampling.sample_random_list(preds)
    true_sample_random = dfs_sampling.sample_random_list(outputs)

    model_random_truthmask = [dfs_verification_tester.dfsverify(As[i], model_sample_random[i]) for i in range(len(model_sample_random))]
    correctness_model_random = sum(model_random_truthmask) / len(model_random_truthmask)

    true_random_truthmask = [dfs_verification_tester.dfsverify(As[i], true_sample_random[i]) for i in range(len(true_sample_random))]
    correctness_true_random = sum(true_random_truthmask) / len(true_random_truthmask)

    t2 = time.time()
    print('random checking in (seconds)', t2-t1)
    ##### ARGMAX
    ## remember to convert from jax arrays to lists for easy subsequent methods using .tolist()
    model_sample_argmax = dfs_sampling.sample_argmax_listofdict(preds)
    true_sample_argmax = dfs_sampling.sample_argmax_listofdatapoint(outputs)

    # compute the fraction of trees sampled from model output fulfilling the necessary conditions
    model_argmax_truthmask = [dfs_verification_tester.dfsverify(As[i], model_sample_argmax[i].tolist()) for i in range(len(model_sample_argmax))]
    correctness_model_argmax = sum(model_argmax_truthmask) / len(model_argmax_truthmask)

    # compute the fraction of trees sampled from true distributions fulfilling the necessary conditions
    true_argmax_truthmask = [dfs_verification_tester.dfsverify(As[i], true_sample_argmax[i].tolist()) for i in range(len(true_sample_argmax))]
    correctness_true_argmax = sum(true_argmax_truthmask) / len(true_argmax_truthmask)

    t3 = time.time()
    print('argmax checking in (seconds)', t3-t2)
    ##### UPWARDS
    model_sample_upwards = dfs_sampling.sample_upwards(preds)
    true_sample_upwards = dfs_sampling.sample_upwards(outputs)

    model_upwards_uniques, model_upwards_valids_uniques, model_upwards_valids = dfs_uniqueness_check.check_uniqueness_dfs(As, preds)
    true_upwards_uniques, true_upwards_valids_uniques, true_upwards_valids  = dfs_uniqueness_check.check_uniqueness_dfs(As, outputs)

    model_upwards_truthmask = [dfs_verification_tester.dfsverify(As[i], model_sample_upwards[i].astype(int)) for i in range(len(model_sample_upwards))]
    correctness_model_upwards = sum(model_upwards_truthmask) / len(model_upwards_truthmask)

    true_upwards_truthmask = [dfs_verification_tester.dfsverify(As[i], true_sample_upwards[i].astype(int)) for i in range(len(true_sample_upwards))]
    correctness_true_upwards = sum(true_upwards_truthmask) / len(true_upwards_truthmask)

    t4 = time.time()
    print('upwards checking in (seconds)', t4 - t3)
    ##### ALTUPWARDS
    model_sample_altUpwards = dfs_sampling.sample_altUpwards(preds)
    true_sample_altUpwards = dfs_sampling.sample_altUpwards(outputs)

    model_altupwards_uniques, model_altupwards_valids_uniques, model_altupwards_valids = dfs_uniqueness_check.check_uniqueness_dfs(As, preds, method="altupwards") #np.zeros(len(true_argmax_truthmask)),np.zeros(len(true_argmax_truthmask)),np.zeros(len(true_argmax_truthmask))
    true_altupwards_uniques, true_altupwards_valids_uniques, true_altupwards_valids  = dfs_uniqueness_check.check_uniqueness_dfs(As, outputs, method="altupwards") #np.zeros(len(true_argmax_truthmask)),np.zeros(len(true_argmax_truthmask)),np.zeros(len(true_argmax_truthmask))

    model_altUpwards_truthmask = [dfs_verification_tester.dfsverify(As[i], model_sample_altUpwards[i].astype(int)) for i in
                                  range(len(model_sample_altUpwards))]
    correctness_model_altUpwards = sum(model_altUpwards_truthmask) / len(model_altUpwards_truthmask)

    true_altUpwards_truthmask = [dfs_verification_tester.dfsverify(As[i], true_sample_altUpwards[i].astype(int)) for i in
                                 range(len(true_sample_altUpwards))]
    correctness_true_altUpwards = sum(true_altUpwards_truthmask) / len(true_altUpwards_truthmask)

    t5 = time.time()
    print('altUpwards checking in (seconds)', t5 - t4)
    #breakpoint()
    As = [i.flatten() for i in As]
    result_dict = {"As": As,
                 #
                 "Argmax_Model_Trees": model_sample_argmax,
                 "Argmax_True_Trees": true_sample_argmax,
                 #
                 "Argmax_Model_Mask": model_argmax_truthmask,
                 "Argmax_True_Mask": true_argmax_truthmask,
                 #
                 "Argmax_Model_Accuracy": correctness_model_argmax,
                 "Argmax_True_Accuracy": correctness_true_argmax,
                 #
                 ###
                 #
                 "Random_Model_Trees": model_sample_random,
                 "Random_True_Trees": true_sample_random,
                 #
                 "Random_Model_Mask": model_random_truthmask,
                 "Random_True_Mask": true_random_truthmask,
                 #
                 "Random_Model_Accuracy": correctness_model_random,
                 "Random_True_Accuracy": correctness_true_random,
                 #
                 ###
                 #
                 "Upwards_Model_Trees": model_sample_upwards,
                 "Upwards_True_Trees": true_sample_upwards,
                 #
                 "Upwards_Model_Mask": model_upwards_truthmask,
                 "Upwards_True_Mask": true_upwards_truthmask,
                 #
                 "Upwards_Model_Accuracy": correctness_model_upwards,
                 "Upwards_True_Accuracy": correctness_true_upwards,

                 "Upwards_Model_Uniques": model_upwards_uniques,
                 "Upwards_Model_Valids_Uniques" : model_upwards_valids_uniques,
                 "Upwards_Model_Valids": model_upwards_valids,

                 "Upwards_True_Uniques" : true_upwards_uniques,
                 "Upwards_True_Valids_Uniques": true_upwards_valids_uniques,
                 "Upwards_True_Valids" : true_upwards_valids,
                 #
                 ###
                 #
                 "altUpwards_Model_Trees": model_sample_altUpwards,
                 "altUpwards_True_Trees": true_sample_altUpwards,
                 #
                 "altUpwards_Model_Mask": model_altUpwards_truthmask,
                 "altUpwards_True_Mask": true_altUpwards_truthmask,
                 #
                 "altUpwards_Model_Accuracy": correctness_model_altUpwards,
                 "altUpwards_True_Accuracy": correctness_true_altUpwards,

                 "altUpwards_Model_Uniques": model_altupwards_uniques,
                 "altUpwards_Model_Valids_Uniques": model_altupwards_valids_uniques,
                 "altUpwards_Model_Valids": model_altupwards_valids,

                 "altUpwards_True_Uniques": true_altupwards_uniques,
                 "altUpwards_True_Valids_Uniques": true_altupwards_valids_uniques,
                 "altUpwards_True_Valids": true_altupwards_valids,
                 }
    #breakpoint()
    save_results(result_dict, f"{filename}_DFS")

    # As[0].reshape((np.sqrt(len(lAs[0])).astype(int)), np.sqrt(len(lAs[0])).astype(int))

    preds = _concat(preds, axis=0)
    out = clrs.evaluate(outputs, preds)

    # breakpoint()
    if extras:
        out.update(extras)
    return {k: unpack(v) for k, v in out.items()}
