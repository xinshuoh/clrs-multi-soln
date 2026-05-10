"""Definition for the Bellman-Ford multi-solution algorithm."""

from clrs._src.specs import Location, Stage, Type
from clrs._src.multi_sol.algorithms.bellman_ford import extractors, generator
from clrs._src.multi_sol.core import definitions
from clrs._src.multi_sol.algorithms.common import batch_extractors
from clrs._src.multi_sol.algorithms.bellman_ford import validator as bf_validation
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
    "pi_h": (Stage.HINT, Location.NODE, Type.POINTER),
    "d": (Stage.HINT, Location.NODE, Type.SCALAR),
    "msk": (Stage.HINT, Location.NODE, Type.MASK),
}


def _sample_randomized_bellman_ford_algorithm(adjacency, source_nodes, rng):
  return RANDOMIZED_ALGORITHM.sample_batch(adjacency, source_nodes, rng)


SOLUTION_SPACE = definitions.MultiSolSolutionSpace(
    batch_extractor=batch_extractors.extract_bellman_ford_graph_and_source,
    validation_method=bf_validation.check_valid_bf_paths,
    extraction_methods=(
        definitions.ExtractionMethod(
            "Argmax",
            extractors.extract_argmax,
            extractors.extract_argmax_true,
        ),
        definitions.ExtractionMethod.same_sampler(
            "Random",
            extractors.extract_random,
        ),
        definitions.ExtractionMethod.same_sampler(
            "Beam",
            extractors.extract_beam,
        ),
        definitions.ExtractionMethod.same_sampler(
            "Greedy",
            extractors.extract_greedy,
        ),
    ),
    generator_sampling_source=definitions.GeneratorSamplingSource(
        name="BellmanFord",
        source_name="Algorithm",
        sample_fn=lambda batch, rng: _sample_randomized_bellman_ford_algorithm(
            batch.adjacency, batch.source_nodes, rng),
    ),
)


DEFINITION = definitions.MultiSolAlgorithm(
    algorithm_name="bellman_ford_multi",
    base_algorithm_name="bellman_ford",
    spec=SPEC,
    generator=generator.bellman_ford_multi,
    sampler_class=samplers.BellmanFordMultiSampler,
    training_distribution=TRAINING_DISTRIBUTION,
    solution_space=SOLUTION_SPACE,
)
