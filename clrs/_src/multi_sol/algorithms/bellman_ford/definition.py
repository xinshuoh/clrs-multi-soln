"""Definition for the Bellman-Ford multi-solution algorithm."""

from clrs._src.multi_sol.algorithms.bellman_ford import extractors, generator
from clrs._src.multi_sol.core import definitions
from clrs._src.multi_sol.algorithms.common import batch_extractors
from clrs._src.multi_sol.algorithms.bellman_ford import validator as bf_validation

TRAINING = definitions.MultiSolTrainingConfig(
    num_solutions=20,
    output_name="pi",
    symbolic_sampler=definitions.RandomizedAlgorithm(
        sample_solution=generator.sample_solution,
        uses_source_node=True,
    ),
)

SOLUTION_SPACE = definitions.MultiSolSolutionSpace(
    batch_extractor=batch_extractors.extract_bellman_ford_graph_and_source,
    validator=bf_validation.check_valid_bf_paths,
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
    algorithm_baseline=definitions.AlgorithmBaseline(
        name="BellmanFord",
    ),
)


DEFINITION = definitions.MultiSolAlgorithm(
    name="bellman_ford_multi",
    base_name="bellman_ford",
    training=TRAINING,
    solution_space=SOLUTION_SPACE,
)
