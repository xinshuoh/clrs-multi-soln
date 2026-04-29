"""Definition for the MST-Prim multi-solution algorithm."""

import numpy as np

from clrs._src.specs import Location, Stage, Type
from clrs._src.multi_sol.algorithms.mst_prim import generator
from clrs._src.multi_sol.core import definitions
from clrs._src.multi_sol.data import adapters
from clrs._src.multi_sol.evaluation import batch_evaluation
from clrs._src.multi_sol.evaluation import definition_evaluation
from clrs._src.multi_sol.sampling import dfs as dfs_sampling
from clrs._src.multi_sol.sampling import mst_prim as mst_sampling
from clrs._src.multi_sol.validation import mst_prim as mst_validation
from clrs._src.multi_sol import samplers


SPEC = {
    "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
    "s": (Stage.INPUT, Location.NODE, Type.MASK_ONE),
    "A": (Stage.INPUT, Location.EDGE, Type.SCALAR),
    "adj": (Stage.INPUT, Location.EDGE, Type.MASK),
    "pi": (Stage.OUTPUT, Location.NODE, Type.POINTER_DISTRIBUTION),
    "pi_h": (Stage.HINT, Location.NODE, Type.POINTER),
    "key": (Stage.HINT, Location.NODE, Type.SCALAR),
    "mark": (Stage.HINT, Location.NODE, Type.MASK),
    "in_queue": (Stage.HINT, Location.NODE, Type.MASK),
    "u": (Stage.HINT, Location.NODE, Type.MASK_ONE),
}


def _sample_randomized_prim_algorithm(adjacency, source_nodes, rng):
  return [
      _randomized_prim_tree(np.asarray(graph), int(source), rng)
      for graph, source in zip(adjacency, source_nodes)
  ]


def _randomized_prim_tree(adjacency, source, rng):
  n = adjacency.shape[0]
  key = np.zeros(n)
  mark = np.zeros(n)
  in_queue = np.zeros(n)
  pi = np.arange(n, dtype=int)
  key[source] = 0
  in_queue[source] = 1

  for _ in range(n):
    effective_keys = np.where(in_queue == 1, key, np.inf)
    min_key_val = np.min(effective_keys)
    if np.isinf(min_key_val):
      break
    candidates = np.where(effective_keys == min_key_val)[0]
    u = int(rng.choice(candidates))
    if in_queue[u] == 0:
      break
    mark[u] = 1
    in_queue[u] = 0
    for v in range(n):
      if adjacency[u, v] != 0:
        if mark[v] == 0 and (in_queue[v] == 0 or adjacency[u, v] < key[v]):
          pi[v] = u
          key[v] = adjacency[u, v]
          in_queue[v] = 1
  return pi


SOLUTION_SPACE = definitions.MultiSolSolutionSpace(
    batch_extractor=adapters.extract_mst_prim_graph_and_source,
    validation_method=mst_validation.check_valid_mst_prim_tree,
    extraction_methods=(
        batch_evaluation.SamplingMethod(
            "Argmax",
            lambda data, _batch: dfs_sampling.sample_argmax_listofdict(data),
            lambda data, _batch: dfs_sampling.sample_argmax_listofdatapoint(
                data),
        ),
        batch_evaluation.SamplingMethod(
            "Random",
            lambda data, _batch: dfs_sampling.sample_random_list(data),
            lambda data, _batch: dfs_sampling.sample_random_list(data),
        ),
        batch_evaluation.SamplingMethod(
            "Tree",
            lambda data, batch: mst_sampling.sample_mst_prim_tree(
                batch.adjacency, batch.source_nodes, data),
            lambda data, batch: mst_sampling.sample_mst_prim_tree(
                batch.adjacency, batch.source_nodes, data),
        ),
        batch_evaluation.SamplingMethod(
            "Greedy",
            lambda data, batch: mst_sampling.sample_mst_prim_greedy(
                batch.adjacency, batch.source_nodes, data),
            lambda data, batch: mst_sampling.sample_mst_prim_greedy(
                batch.adjacency, batch.source_nodes, data),
        ),
    ),
    generator_sampling_source=batch_evaluation.AlgorithmSamplingSource(
        "Prim",
        "Algorithm",
        lambda batch, rng: _sample_randomized_prim_algorithm(
            batch.adjacency, batch.source_nodes, rng),
    ),
)


def evaluate_mst_prim_multisol_batch(**kwargs):
  return definition_evaluation.evaluate_definition(
      definition=DEFINITION, **kwargs)


DEFINITION = definitions.MultiSolAlgorithmDefinition(
    algorithm_name="mst_prim_multi",
    base_algorithm_name="mst_prim",
    spec=SPEC,
    sampler_class=samplers.MSTPrimMultiSampler,
    algorithm=generator.mst_prim_multi,
    evaluator=evaluate_mst_prim_multisol_batch,
    solution_space=SOLUTION_SPACE,
)
