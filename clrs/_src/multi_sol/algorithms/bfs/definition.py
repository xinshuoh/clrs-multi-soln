"""Definition for the BFS multi-solution algorithm."""

import numpy as np

from clrs._src.specs import Location, Stage, Type
from clrs._src.multi_sol.algorithms.bfs import generator
from clrs._src.multi_sol.core import definitions
from clrs._src.multi_sol.data import adapters
from clrs._src.multi_sol.evaluation import batch_evaluation
from clrs._src.multi_sol.evaluation import definition_evaluation
from clrs._src.multi_sol.sampling import bfs as bfs_sampling
from clrs._src.multi_sol.sampling import dfs as dfs_sampling
from clrs._src.multi_sol.validation import bfs as bfs_validation
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
    "reach_h": (Stage.HINT, Location.NODE, Type.MASK),
    "pi_h": (Stage.HINT, Location.NODE, Type.POINTER),
}


def _sample_randomized_bfs_algorithm(adjacency, source_nodes, rng):
  return [
      _randomized_bfs_tree(np.asarray(graph), int(source), rng)
      for graph, source in zip(adjacency, source_nodes)
  ]


def _randomized_bfs_tree(adjacency, source, rng):
  n = adjacency.shape[0]
  reach = np.zeros(n, dtype=bool)
  pi = np.arange(n, dtype=int)
  reach[source] = True

  while True:
    prev_reach = np.copy(reach)
    sources = np.where(prev_reach)[0]
    rng.shuffle(sources)
    for src in sources:
      for child in range(n):
        if adjacency[src, child] > 0:
          if pi[child] == child and child != source:
            pi[child] = int(src)
          reach[child] = True
    if np.all(reach == prev_reach):
      break
  return pi


SOLUTION_SPACE = definitions.MultiSolSolutionSpace(
    batch_extractor=adapters.extract_bfs_graph_and_source,
    validation_method=bfs_validation.check_valid_bfs_tree,
    extraction_methods=(
        batch_evaluation.SamplingMethod(
            "Categorical",
            lambda data, _batch: bfs_sampling.sample_bfs_categorical(data),
            lambda data, _batch: bfs_sampling.sample_bfs_categorical(data),
        ),
        batch_evaluation.SamplingMethod(
            "Random",
            lambda data, _batch: dfs_sampling.sample_random_list(data),
            lambda data, _batch: dfs_sampling.sample_random_list(data),
        ),
        batch_evaluation.SamplingMethod(
            "Prim",
            lambda data, batch: bfs_sampling.sample_bfs_prim(
                data, batch.source_nodes),
            lambda data, batch: bfs_sampling.sample_bfs_prim(
                data, batch.source_nodes),
        ),
        batch_evaluation.SamplingMethod(
            "Beam",
            lambda data, batch: bfs_sampling.sample_bfs_beam(
                data, batch.source_nodes, beam_width=3),
            lambda data, batch: bfs_sampling.sample_bfs_beam(
                data, batch.source_nodes, beam_width=3),
        ),
    ),
    generator_sampling_source=batch_evaluation.AlgorithmSamplingSource(
        "BFS",
        "Algorithm",
        lambda batch, rng: _sample_randomized_bfs_algorithm(
            batch.adjacency, batch.source_nodes, rng),
    ),
    include_source_nodes=True,
)


def evaluate_bfs_multisol_batch(**kwargs):
  return definition_evaluation.evaluate_definition(
      definition=DEFINITION, **kwargs)


DEFINITION = definitions.MultiSolAlgorithmDefinition(
    algorithm_name="bfs_multi",
    base_algorithm_name="bfs",
    spec=SPEC,
    sampler_class=samplers.BfsMultiSampler,
    algorithm=generator.bfs_multi,
    evaluator=evaluate_bfs_multisol_batch,
    solution_space=SOLUTION_SPACE,
)
