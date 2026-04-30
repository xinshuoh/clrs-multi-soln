"""Definition for the BFS multi-solution algorithm."""

from clrs._src.specs import Location, Stage, Type
from clrs._src.multi_sol.algorithms import graphs
from clrs._src.multi_sol.algorithms.bfs import generator
from clrs._src.multi_sol.core import definitions
from clrs._src.multi_sol.data import adapters
from clrs._src.multi_sol.evaluation import definition_evaluation
from clrs._src.multi_sol.sampling import bfs as bfs_sampling
from clrs._src.multi_sol.sampling import dfs as dfs_sampling
from clrs._src.multi_sol.validation import bfs as bfs_validation
from clrs._src.multi_sol import samplers


TRAINING_DISTRIBUTION = definitions.TrainingDistribution(
    num_solutions=20,
    output_name="pi",
)

RANDOMIZED_ALGORITHM = definitions.RandomizedAlgorithm(
    sample_solution=graphs.bfs_multi,
    uses_source_node=True,
)


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
  return RANDOMIZED_ALGORITHM.sample_batch(adjacency, source_nodes, rng)


SOLUTION_SPACE = definitions.MultiSolSolutionSpace(
    batch_extractor=adapters.extract_bfs_graph_and_source,
    validation_method=bfs_validation.check_valid_bfs_tree,
    extraction_methods=(
        definitions.ExtractionMethod.same_sampler(
            "Categorical",
            lambda data, _batch: bfs_sampling.sample_bfs_categorical(data),
        ),
        definitions.ExtractionMethod.same_sampler(
            "Random",
            lambda data, _batch: dfs_sampling.sample_random_list(data),
        ),
        definitions.ExtractionMethod.same_sampler(
            "Prim",
            lambda data, batch: bfs_sampling.sample_bfs_prim(
                data, batch.source_nodes),
        ),
        definitions.ExtractionMethod.same_sampler(
            "Beam",
            lambda data, batch: bfs_sampling.sample_bfs_beam(
                data, batch.source_nodes, beam_width=3),
        ),
    ),
    generator_sampling_source=definitions.GeneratorSamplingSource(
        name="BFS",
        source_name="Algorithm",
        sample_fn=lambda batch, rng: _sample_randomized_bfs_algorithm(
            batch.adjacency, batch.source_nodes, rng),
    ),
    include_source_nodes=True,
)


def evaluate_bfs_multisol_batch(**kwargs):
  return definition_evaluation.evaluate_definition(
      definition=DEFINITION, **kwargs)


def algorithm_spec():
  return SPEC


def training_distribution():
  return TRAINING_DISTRIBUTION


DEFINITION = definitions.MultiSolAlgorithm(
    algorithm_name="bfs_multi",
    base_algorithm_name="bfs",
    spec=SPEC,
    sampler_class=samplers.BfsMultiSampler,
    algorithm=generator.bfs_multi,
    evaluator=evaluate_bfs_multisol_batch,
    training_distribution=TRAINING_DISTRIBUTION,
    randomized_algorithm=RANDOMIZED_ALGORITHM,
    solution_space=SOLUTION_SPACE,
)
