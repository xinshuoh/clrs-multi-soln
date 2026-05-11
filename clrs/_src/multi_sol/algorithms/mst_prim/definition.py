"""Definition for the MST-Prim multi-solution algorithm."""

from clrs._src.multi_sol.algorithms.mst_prim import extractors, generator, validator
from clrs._src.multi_sol.algorithms.common import batch_extractors
from clrs._src.multi_sol.interfaces import Generator, MultiSolAlgorithm, ReferenceSampler

ALGORITHM = MultiSolAlgorithm(
    name="mst_prim_multi",
    base_name="mst_prim",
    generator=Generator(
        sample_target=generator.mst_prim_multi,
        sample_solution=generator.sample_solution,
        num_solutions=20,
        output_name="pi",
        uses_source_node=True,
    ),
    batch_extractor=batch_extractors.extract_mst_prim_graph_and_source,
    validator=validator.check_valid_mst_prim_tree,
    extractors=extractors.EXTRACTORS,
    reference_sampler=ReferenceSampler(
        name="Prim",
    ),
)
