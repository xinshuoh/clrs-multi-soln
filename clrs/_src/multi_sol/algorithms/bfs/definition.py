"""Definition for the BFS multi-solution algorithm."""

from clrs._src.multi_sol.algorithms.bfs import extractors, generator, validator
from clrs._src.multi_sol.core import definitions
from clrs._src.multi_sol.algorithms.common import batch_extractors

TRAINING = definitions.MultiSolTrainingConfig(
    num_solutions=20,
    output_name="pi",
    symbolic_sampler=definitions.RandomizedAlgorithm(
        sample_solution=generator.sample_solution,
        uses_source_node=True,
    ),
)

SOLUTION_SPACE = definitions.MultiSolSolutionSpace(
    batch_extractor=batch_extractors.extract_bfs_graph_and_source,
    validator=validator.check_valid_bfs_tree,
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
    algorithm_baseline=definitions.AlgorithmBaseline(
        name="BFS",
    ),
    include_source_nodes=True,
)


DEFINITION = definitions.MultiSolAlgorithm(
    name="bfs_multi",
    base_name="bfs",
    training=TRAINING,
    solution_space=SOLUTION_SPACE,
)
