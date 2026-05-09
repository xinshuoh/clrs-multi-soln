"""Definition for the BFS multi-solution algorithm."""

from clrs._src.specs import Location, Stage, Type
from clrs._src.multi_sol.algorithms.bfs import extractors, generator, validator
from clrs._src.multi_sol.core import definitions
from clrs._src.multi_sol.evaluation import adapters
from clrs._src.multi_sol.evaluation import definition_evaluation
from clrs._src import samplers

TRAINING_DISTRIBUTION = definitions.TrainingDistribution(
    num_solutions=20,
    output_name="pi",
)

RANDOMIZED_ALGORITHM = definitions.RandomizedAlgorithm(
    sample_solution=generator.sample_solution,
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
    validation_method=validator.check_valid_bfs_tree,
    extraction_methods=(
        definitions.ExtractionMethod.same_sampler(
            "Categorical",
            extractors.extract_categorical,
        ),
        definitions.ExtractionMethod.same_sampler(
            "Random",
            extractors.extract_random,
        ),
        definitions.ExtractionMethod.same_sampler(
            "Prim",
            extractors.extract_prim,
        ),
        definitions.ExtractionMethod.same_sampler(
            "Beam",
            extractors.extract_beam,
        ),
    ),
    generator_sampling_source=definitions.GeneratorSamplingSource(
        name="BFS",
        source_name="Algorithm",
        sample_fn=lambda batch, rng: _sample_randomized_bfs_algorithm(batch.adjacency, batch.
                                                                      source_nodes, rng),
    ),
    include_source_nodes=True,
)


def evaluate_bfs_multisol_batch(**kwargs):
  return definition_evaluation.evaluate_definition(definition=DEFINITION, **kwargs)


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
    generator=generator.bfs_multi,
    evaluator=evaluate_bfs_multisol_batch,
    extractors=extractors.EXTRACTORS,
    validator=validator.check_valid_bfs_tree,
    training_distribution=TRAINING_DISTRIBUTION,
    randomized_algorithm=RANDOMIZED_ALGORITHM,
    solution_space=SOLUTION_SPACE,
)
