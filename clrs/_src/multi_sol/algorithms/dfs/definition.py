"""Definition for the DFS multi-solution algorithm."""

import numpy as np

from clrs._src.specs import Location, Stage, Type
from clrs._src.multi_sol.algorithms.dfs import generator
from clrs._src.multi_sol.core import definitions
from clrs._src.multi_sol.data import adapters
from clrs._src.multi_sol.evaluation import batch_evaluation
from clrs._src.multi_sol.evaluation import definition_evaluation
from clrs._src.multi_sol.sampling import dfs as dfs_sampling
from clrs._src.multi_sol.validation import dfs as dfs_validation
from clrs._src.multi_sol import samplers


SPEC = {
    "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
    "A": (Stage.INPUT, Location.EDGE, Type.SCALAR),
    "adj": (Stage.INPUT, Location.EDGE, Type.MASK),
    "pi": (
        Stage.OUTPUT,
        Location.NODE,
        Type.POINTER_DISTRIBUTION,
    ),
    "pi_h": (Stage.HINT, Location.NODE, Type.POINTER),
    "color": (Stage.HINT, Location.NODE, Type.CATEGORICAL),
    "d": (Stage.HINT, Location.NODE, Type.SCALAR),
    "f": (Stage.HINT, Location.NODE, Type.SCALAR),
    "s_prev": (Stage.HINT, Location.NODE, Type.POINTER),
    "s": (Stage.HINT, Location.NODE, Type.MASK_ONE),
    "u": (Stage.HINT, Location.NODE, Type.MASK_ONE),
    "v": (Stage.HINT, Location.NODE, Type.MASK_ONE),
    "s_last": (Stage.HINT, Location.NODE, Type.MASK_ONE),
    "time": (Stage.HINT, Location.GRAPH, Type.SCALAR),
}


def _validate_dfs_tree(adjacency, parent_tree, _source):
  return dfs_validation.check_valid_dfsTree(adjacency, parent_tree)


def _sample_randomized_dfs_algorithm(adjacency, rng):
  return [
      _randomized_dfs_tree(np.asarray(graph), rng)
      for graph in adjacency
  ]


def _randomized_dfs_tree(adjacency, rng):
  n = adjacency.shape[0]
  color = np.zeros(n, dtype=np.int32)
  pi = np.arange(n, dtype=int)
  s_prev = np.arange(n, dtype=int)
  shuffled = rng.permutation(n)

  for s in range(n):
    if color[s] != 0:
      continue
    s_last = s
    u = s
    while True:
      if color[u] == 0:
        color[u] = 1

      for v in shuffled:
        if adjacency[u, v] != 0 and color[v] == 0:
          pi[v] = u
          color[v] = 1
          s_prev[v] = s_last
          s_last = v
          break

      if s_last == u:
        color[u] = 2
        if s_prev[u] == u:
          break
        parent = s_prev[s_last]
        s_prev[s_last] = s_last
        s_last = parent

      u = s_last
  return pi


SOLUTION_SPACE = definitions.MultiSolSolutionSpace(
    batch_extractor=adapters.extract_dfs_graph_and_source,
    validation_method=_validate_dfs_tree,
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
            "Upwards",
            lambda data, _batch: dfs_sampling.sample_upwards(data),
            lambda data, _batch: dfs_sampling.sample_upwards(data),
        ),
        batch_evaluation.SamplingMethod(
            "altUpwards",
            lambda data, _batch: dfs_sampling.sample_altUpwards(data),
            lambda data, _batch: dfs_sampling.sample_altUpwards(data),
        ),
    ),
    generator_sampling_source=batch_evaluation.AlgorithmSamplingSource(
        "DFS",
        "Algorithm",
        lambda batch, rng: _sample_randomized_dfs_algorithm(batch.adjacency, rng),
    ),
)


def evaluate_dfs_multisol_batch(**kwargs):
  return definition_evaluation.evaluate_definition(
      definition=DEFINITION, **kwargs)


DEFINITION = definitions.MultiSolAlgorithmDefinition(
    algorithm_name="dfs_multi",
    base_algorithm_name="dfs",
    spec=SPEC,
    sampler_class=samplers.DfsMultiSampler,
    algorithm=generator.dfs_multi,
    evaluator=evaluate_dfs_multisol_batch,
    solution_space=SOLUTION_SPACE,
)
