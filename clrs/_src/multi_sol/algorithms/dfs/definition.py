"""Definition for the DFS multi-solution algorithm."""

from clrs._src.specs import Location, Stage, Type
from clrs._src.multi_sol.algorithms.dfs import extractors, generator, validator
from clrs._src.multi_sol.core import definitions
from clrs._src.multi_sol.algorithms.common import batch_extractors
from clrs._src import samplers

TRAINING_DISTRIBUTION = definitions.TrainingDistribution(
    num_solutions=20,
    output_name="pi",
)

RANDOMIZED_ALGORITHM = definitions.RandomizedAlgorithm(sample_solution=generator.sample_solution,)

SPEC = {
    "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
    "A": (Stage.INPUT, Location.EDGE, Type.SCALAR),
    "adj": (Stage.INPUT, Location.EDGE, Type.MASK),
    "pi": (
        Stage.OUTPUT,
        Location.NODE,
        Type.POINTER_DISTRIBUTION,
    ),
    "pi_h": (Stage.HINT, Location.NODE, Type.POINTER),
    "color": (Stage.HINT, Location.NODE, Type.CATEGORICAL),
    "d": (Stage.HINT, Location.NODE, Type.SCALAR),
    "f": (Stage.HINT, Location.NODE, Type.SCALAR),
    "s_prev": (Stage.HINT, Location.NODE, Type.POINTER),
    "s": (Stage.HINT, Location.NODE, Type.MASK_ONE),
    "u": (Stage.HINT, Location.NODE, Type.MASK_ONE),
    "v": (Stage.HINT, Location.NODE, Type.MASK_ONE),
    "s_last": (Stage.HINT, Location.NODE, Type.MASK_ONE),
    "time": (Stage.HINT, Location.GRAPH, Type.SCALAR),
}


def _validate_dfs_tree(adjacency, parent_tree, _source):
  return validator.check_valid_dfs_tree(adjacency, parent_tree)


def _sample_randomized_dfs_algorithm(adjacency, rng):
  return RANDOMIZED_ALGORITHM.sample_batch(adjacency, None, rng)


SOLUTION_SPACE = definitions.MultiSolSolutionSpace(
    batch_extractor=batch_extractors.extract_dfs_graph_and_source,
    validation_method=_validate_dfs_tree,
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
    generator_sampling_source=definitions.GeneratorSamplingSource(
        name="DFS",
        source_name="Algorithm",
        sample_fn=lambda batch, rng: _sample_randomized_dfs_algorithm(batch.adjacency, rng),
    ),
)


DEFINITION = definitions.MultiSolAlgorithm(
    algorithm_name="dfs_multi",
    base_algorithm_name="dfs",
    spec=SPEC,
    generator=generator.dfs_multi,
    sampler_class=samplers.DfsMultiSampler,
    training_distribution=TRAINING_DISTRIBUTION,
    solution_space=SOLUTION_SPACE,
)
