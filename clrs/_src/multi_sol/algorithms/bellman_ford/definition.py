"""Definition for the Bellman-Ford multi-solution algorithm."""

import numpy as np

from clrs._src.specs import Location, Stage, Type
from clrs._src.multi_sol.algorithms.bellman_ford import generator
from clrs._src.multi_sol.core import definitions
from clrs._src.multi_sol.data import adapters
from clrs._src.multi_sol.evaluation import batch_evaluation
from clrs._src.multi_sol.evaluation import definition_evaluation
from clrs._src.multi_sol.sampling import bellman_ford as bf_sampling
from clrs._src.multi_sol.sampling import dfs as dfs_sampling
from clrs._src.multi_sol.validation import bellman_ford as bf_validation
from clrs._src.multi_sol import samplers


SPEC = {
    "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
    "s": (Stage.INPUT, Location.NODE, Type.MASK_ONE),
    "A": (Stage.INPUT, Location.EDGE, Type.SCALAR),
    "adj": (Stage.INPUT, Location.EDGE, Type.MASK),
    "pi": (
        Stage.OUTPUT,
        Location.NODE,
        Type.POINTER_DISTRIBUTION,
    ),
    "pi_h": (Stage.HINT, Location.NODE, Type.POINTER),
    "d": (Stage.HINT, Location.NODE, Type.SCALAR),
    "msk": (Stage.HINT, Location.NODE, Type.MASK),
}


def _sample_randomized_bellman_ford_algorithm(adjacency, source_nodes, rng):
  return [
      _randomized_bellman_ford_tree(np.asarray(graph), int(source), rng)
      for graph, source in zip(adjacency, source_nodes)
  ]


def _randomized_bellman_ford_tree(adjacency, source, rng):
  n = adjacency.shape[0]
  d = np.zeros(n)
  pi = np.arange(n, dtype=int)
  msk = np.zeros(n)
  d[source] = 0
  msk[source] = 1

  shuffled_sources = rng.permutation(n)
  shuffled_targets = rng.permutation(n)
  while True:
    prev_d = np.copy(d)
    prev_msk = np.copy(msk)
    for u in shuffled_sources:
      for v in shuffled_targets:
        if prev_msk[u] == 1 and adjacency[u, v] != 0:
          if msk[v] == 0 or prev_d[u] + adjacency[u, v] < d[v]:
            d[v] = prev_d[u] + adjacency[u, v]
            pi[v] = int(u)
          msk[v] = 1
    if np.all(d == prev_d):
      break
  return pi


SOLUTION_SPACE = definitions.MultiSolSolutionSpace(
    batch_extractor=adapters.extract_bellman_ford_graph_and_source,
    validation_method=bf_validation.check_valid_bf_paths,
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
            "Beam",
            lambda data, batch: bf_sampling.sample_beamsearch(
                batch.adjacency, batch.source_nodes, data),
            lambda data, batch: bf_sampling.sample_beamsearch(
                batch.adjacency, batch.source_nodes, data),
        ),
        batch_evaluation.SamplingMethod(
            "Greedy",
            lambda data, batch: bf_sampling.sample_greedysearch(
                batch.adjacency, batch.source_nodes, data),
            lambda data, batch: bf_sampling.sample_greedysearch(
                batch.adjacency, batch.source_nodes, data),
        ),
    ),
    generator_sampling_source=batch_evaluation.AlgorithmSamplingSource(
        "BellmanFord",
        "Algorithm",
        lambda batch, rng: _sample_randomized_bellman_ford_algorithm(
            batch.adjacency, batch.source_nodes, rng),
    ),
)


def evaluate_bf_multisol_batch(**kwargs):
  return definition_evaluation.evaluate_definition(
      definition=DEFINITION, **kwargs)


DEFINITION = definitions.MultiSolAlgorithmDefinition(
    algorithm_name="bellman_ford_multi",
    base_algorithm_name="bellman_ford",
    spec=SPEC,
    sampler_class=samplers.BellmanFordMultiSampler,
    algorithm=generator.bellman_ford_multi,
    evaluator=evaluate_bf_multisol_batch,
    solution_space=SOLUTION_SPACE,
)
