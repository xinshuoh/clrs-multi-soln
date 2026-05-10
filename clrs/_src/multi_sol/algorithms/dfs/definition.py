"""Definition for the DFS multi-solution algorithm."""

from clrs._src.multi_sol.algorithms.dfs import extractors, generator, validator
from clrs._src.multi_sol.core import definitions
from clrs._src.multi_sol.algorithms.common import batch_extractors

TRAINING = definitions.MultiSolTrainingConfig(
    num_solutions=20,
    output_name="pi",
    symbolic_sampler=definitions.RandomizedAlgorithm(
        sample_solution=generator.sample_solution,
    ),
)


def _validate_dfs_tree(adjacency, parent_tree, _source):
  return validator.check_valid_dfs_tree(adjacency, parent_tree)

SOLUTION_SPACE = definitions.MultiSolSolutionSpace(
    batch_extractor=batch_extractors.extract_dfs_graph_and_source,
    validator=_validate_dfs_tree,
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
            "Upwards",
            extractors.extract_upwards,
        ),
        definitions.ExtractionMethod.same_sampler(
            "altUpwards",
            extractors.extract_alt_upwards,
        ),
    ),
    algorithm_baseline=definitions.AlgorithmBaseline(
        name="DFS",
    ),
)


DEFINITION = definitions.MultiSolAlgorithm(
    name="dfs_multi",
    base_name="dfs",
    training=TRAINING,
    solution_space=SOLUTION_SPACE,
)
