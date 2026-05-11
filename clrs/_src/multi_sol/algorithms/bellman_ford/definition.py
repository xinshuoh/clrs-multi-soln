"""Definition for the Bellman-Ford multi-solution algorithm."""

from clrs._src.multi_sol.algorithms.bellman_ford import extractors, generator
from clrs._src.multi_sol.algorithms.common import batch_extractors
from clrs._src.multi_sol.algorithms.bellman_ford import validator as bf_validation
from clrs._src.multi_sol.interfaces import Generator, MultiSolAlgorithm, ReferenceSampler

ALGORITHM = MultiSolAlgorithm(
    name="bellman_ford_multi",
    base_name="bellman_ford",
    generator=Generator(
        sample_target=generator.bellman_ford_multi,
        sample_solution=generator.sample_solution,
        num_solutions=20,
        output_name="pi",
        uses_source_node=True,
    ),
    batch_extractor=batch_extractors.extract_bellman_ford_graph_and_source,
    validator=bf_validation.check_valid_bf_paths,
    extractors=extractors.EXTRACTORS,
    reference_sampler=ReferenceSampler(
        name="BellmanFord",
    ),
)
