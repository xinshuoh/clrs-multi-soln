"""Definition for the MST-Prim multi-solution algorithm."""

from clrs._src.multi_sol.algorithms.mst_prim import extractors, generator, validator
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
    batch_extractor=batch_extractors.extract_mst_prim_graph_and_source,
    validator=validator.check_valid_mst_prim_tree,
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
            "Tree",
            extractors.extract_tree,
        ),
        definitions.ExtractionMethod.same_sampler(
            "Greedy",
            extractors.extract_greedy,
        ),
    ),
    algorithm_baseline=definitions.AlgorithmBaseline(
        name="Prim",
    ),
)


DEFINITION = definitions.MultiSolAlgorithm(
    name="mst_prim_multi",
    base_name="mst_prim",
    training=TRAINING,
    solution_space=SOLUTION_SPACE,
)
