"""Definition for the Bellman-Ford multi-solution algorithm."""

from clrs._src.specs import Location, Stage, Type
from clrs._src.multi_sol.algorithms.bellman_ford import generator
from clrs._src.multi_sol.core import definitions
from clrs._src.multi_sol.data import adapters
from clrs._src.multi_sol.evaluation import definition_evaluation
from clrs._src.multi_sol.sampling import bellman_ford as bf_sampling
from clrs._src.multi_sol.sampling import dfs as dfs_sampling
from clrs._src.multi_sol.algorithms.bellman_ford import validator as bf_validation
from clrs._src.multi_sol import samplers


TRAINING_DISTRIBUTION = definitions.TrainingDistribution(
    num_solutions=20,
    output_name="pi",
)

RANDOMIZED_ALGORITHM = definitions.RandomizedAlgorithm(
    sample_solution=generator.sample_solution,
    uses_source_node=True,
)


SPEC = {
    "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
    "s": (Stage.INPUT, Location.NODE, Type.MASK_ONE),
    "A": (Stage.INPUT, Location.EDGE, Type.SCALAR),
    "adj": (Stage.INPUT, Location.EDGE, Type.MASK),
    "pi": (
        Stage.OUTPUT,
        Location.NODE,
        Type.POINTER_DISTRIBUTION,
    ),
    "pi_h": (Stage.HINT, Location.NODE, Type.POINTER),
    "d": (Stage.HINT, Location.NODE, Type.SCALAR),
    "msk": (Stage.HINT, Location.NODE, Type.MASK),
}


def _sample_randomized_bellman_ford_algorithm(adjacency, source_nodes, rng):
  return RANDOMIZED_ALGORITHM.sample_batch(adjacency, source_nodes, rng)


SOLUTION_SPACE = definitions.MultiSolSolutionSpace(
    batch_extractor=adapters.extract_bellman_ford_graph_and_source,
    validation_method=bf_validation.check_valid_bf_paths,
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
            "Beam",
            lambda data, batch: bf_sampling.sample_beamsearch(
                batch.adjacency, batch.source_nodes, data),
        ),
        definitions.ExtractionMethod.same_sampler(
            "Greedy",
            lambda data, batch: bf_sampling.sample_greedysearch(
                batch.adjacency, batch.source_nodes, data),
        ),
    ),
    generator_sampling_source=definitions.GeneratorSamplingSource(
        name="BellmanFord",
        source_name="Algorithm",
        sample_fn=lambda batch, rng: _sample_randomized_bellman_ford_algorithm(
            batch.adjacency, batch.source_nodes, rng),
    ),
)


def evaluate_bf_multisol_batch(**kwargs):
  return definition_evaluation.evaluate_definition(
      definition=DEFINITION, **kwargs)


def algorithm_spec():
  return SPEC


def training_distribution():
  return TRAINING_DISTRIBUTION


DEFINITION = definitions.MultiSolAlgorithm(
    algorithm_name="bellman_ford_multi",
    base_algorithm_name="bellman_ford",
    spec=SPEC,
    sampler_class=samplers.BellmanFordMultiSampler,
    algorithm=generator.bellman_ford_multi,
    evaluator=evaluate_bf_multisol_batch,
    training_distribution=TRAINING_DISTRIBUTION,
    randomized_algorithm=RANDOMIZED_ALGORITHM,
    solution_space=SOLUTION_SPACE,
)
