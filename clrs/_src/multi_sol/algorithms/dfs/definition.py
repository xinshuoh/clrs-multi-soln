"""Definition for the DFS multi-solution algorithm."""

from clrs._src.specs import Location, Stage, Type
from clrs._src.multi_sol.algorithms import graphs
from clrs._src.multi_sol.algorithms.dfs import generator
from clrs._src.multi_sol.core import definitions
from clrs._src.multi_sol.data import adapters
from clrs._src.multi_sol.evaluation import definition_evaluation
from clrs._src.multi_sol.sampling import dfs as dfs_sampling
from clrs._src.multi_sol.validation import dfs as dfs_validation
from clrs._src.multi_sol import samplers


TRAINING_DISTRIBUTION = definitions.TrainingDistribution(
    num_solutions=20,
    output_name="pi",
)

RANDOMIZED_ALGORITHM = definitions.RandomizedAlgorithm(
    sample_solution=graphs.dfs_multi,
)


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
  return dfs_validation.check_valid_dfsTree(adjacency, parent_tree)


def _sample_randomized_dfs_algorithm(adjacency, rng):
  return RANDOMIZED_ALGORITHM.sample_batch(adjacency, None, rng)


SOLUTION_SPACE = definitions.MultiSolSolutionSpace(
    batch_extractor=adapters.extract_dfs_graph_and_source,
    validation_method=_validate_dfs_tree,
    extraction_methods=(
        definitions.ExtractionMethod(
            "Argmax",
            lambda data, _batch: dfs_sampling.sample_argmax_listofdict(data),
            lambda data, _batch: dfs_sampling.sample_argmax_listofdatapoint(
                data),
        ),
        definitions.ExtractionMethod.same_sampler(
            "Random",
            lambda data, _batch: dfs_sampling.sample_random_list(data),
        ),
        definitions.ExtractionMethod.same_sampler(
            "Upwards",
            lambda data, _batch: dfs_sampling.sample_upwards(data),
        ),
        definitions.ExtractionMethod.same_sampler(
            "altUpwards",
            lambda data, _batch: dfs_sampling.sample_altUpwards(data),
        ),
    ),
    generator_sampling_source=definitions.GeneratorSamplingSource(
        name="DFS",
        source_name="Algorithm",
        sample_fn=lambda batch, rng: _sample_randomized_dfs_algorithm(
            batch.adjacency, rng),
    ),
)


def evaluate_dfs_multisol_batch(**kwargs):
  return definition_evaluation.evaluate_definition(
      definition=DEFINITION, **kwargs)


def algorithm_spec():
  return SPEC


def training_distribution():
  return TRAINING_DISTRIBUTION


DEFINITION = definitions.MultiSolAlgorithm(
    algorithm_name="dfs_multi",
    base_algorithm_name="dfs",
    spec=SPEC,
    sampler_class=samplers.DfsMultiSampler,
    algorithm=generator.dfs_multi,
    evaluator=evaluate_dfs_multisol_batch,
    training_distribution=TRAINING_DISTRIBUTION,
    randomized_algorithm=RANDOMIZED_ALGORITHM,
    solution_space=SOLUTION_SPACE,
)
