"""Definition for the DFS multi-solution algorithm."""

from clrs._src.multi_sol.algorithms.dfs import extractors, generator, validator
from clrs._src.multi_sol.algorithms.common import batch_extractors
from clrs._src.multi_sol.interfaces import Generator, MultiSolAlgorithm, ReferenceSampler

def _validate_dfs_tree(adjacency, parent_tree, _source):
  return validator.check_valid_dfs_tree(adjacency, parent_tree)

ALGORITHM = MultiSolAlgorithm(
    name="dfs_multi",
    base_name="dfs",
    generator=Generator(
        sample_target=generator.dfs_multi,
        sample_solution=generator.sample_solution,
        num_solutions=20,
        output_name="pi",
    ),
    batch_extractor=batch_extractors.extract_dfs_graph_and_source,
    validator=_validate_dfs_tree,
    extractors=extractors.EXTRACTORS,
    reference_sampler=ReferenceSampler(
        name="DFS",
    ),
)
