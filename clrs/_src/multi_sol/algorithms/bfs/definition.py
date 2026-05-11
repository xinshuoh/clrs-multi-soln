"""Definition for the BFS multi-solution algorithm."""

from clrs._src.multi_sol.algorithms.bfs import extractors, generator, validator
from clrs._src.multi_sol.algorithms.common import batch_extractors
from clrs._src.multi_sol.interfaces import Generator, MultiSolAlgorithm, ReferenceSampler

ALGORITHM = MultiSolAlgorithm(
    name="bfs_multi",
    base_name="bfs",
    generator=Generator(
        sample_target=generator.bfs_multi,
        sample_solution=generator.sample_solution,
        num_solutions=20,
        output_name="pi",
        uses_source_node=True,
    ),
    batch_extractor=batch_extractors.extract_bfs_graph_and_source,
    validator=validator.check_valid_bfs_tree,
    extractors=extractors.EXTRACTORS,
    reference_sampler=ReferenceSampler(
        name="BFS",
    ),
    include_source_nodes_in_report=True,
)
