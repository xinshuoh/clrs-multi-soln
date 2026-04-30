"""Definition for the MST-Prim multi-solution algorithm."""

from clrs._src.specs import Location, Stage, Type
from clrs._src.multi_sol.algorithms import graphs
from clrs._src.multi_sol.algorithms.mst_prim import generator
from clrs._src.multi_sol.core import definitions
from clrs._src.multi_sol.data import adapters
from clrs._src.multi_sol.evaluation import definition_evaluation
from clrs._src.multi_sol.sampling import dfs as dfs_sampling
from clrs._src.multi_sol.sampling import mst_prim as mst_sampling
from clrs._src.multi_sol.validation import mst_prim as mst_validation
from clrs._src.multi_sol import samplers


TRAINING_DISTRIBUTION = definitions.TrainingDistribution(
    num_solutions=20,
    output_name="pi",
)

RANDOMIZED_ALGORITHM = definitions.RandomizedAlgorithm(
    sample_solution=graphs.mst_prim_multi,
    uses_source_node=True,
)


SPEC = {
    "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
    "s": (Stage.INPUT, Location.NODE, Type.MASK_ONE),
    "A": (Stage.INPUT, Location.EDGE, Type.SCALAR),
    "adj": (Stage.INPUT, Location.EDGE, Type.MASK),
    "pi": (Stage.OUTPUT, Location.NODE, Type.POINTER_DISTRIBUTION),
    "pi_h": (Stage.HINT, Location.NODE, Type.POINTER),
    "key": (Stage.HINT, Location.NODE, Type.SCALAR),
    "mark": (Stage.HINT, Location.NODE, Type.MASK),
    "in_queue": (Stage.HINT, Location.NODE, Type.MASK),
    "u": (Stage.HINT, Location.NODE, Type.MASK_ONE),
}


def _sample_randomized_prim_algorithm(adjacency, source_nodes, rng):
  return RANDOMIZED_ALGORITHM.sample_batch(adjacency, source_nodes, rng)


SOLUTION_SPACE = definitions.MultiSolSolutionSpace(
    batch_extractor=adapters.extract_mst_prim_graph_and_source,
    validation_method=mst_validation.check_valid_mst_prim_tree,
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
            "Tree",
            lambda data, batch: mst_sampling.sample_mst_prim_tree(
                batch.adjacency, batch.source_nodes, data),
        ),
        definitions.ExtractionMethod.same_sampler(
            "Greedy",
            lambda data, batch: mst_sampling.sample_mst_prim_greedy(
                batch.adjacency, batch.source_nodes, data),
        ),
    ),
    generator_sampling_source=definitions.GeneratorSamplingSource(
        name="Prim",
        source_name="Algorithm",
        sample_fn=lambda batch, rng: _sample_randomized_prim_algorithm(
            batch.adjacency, batch.source_nodes, rng),
    ),
)


def evaluate_mst_prim_multisol_batch(**kwargs):
  return definition_evaluation.evaluate_definition(
      definition=DEFINITION, **kwargs)


def algorithm_spec():
  return SPEC


def training_distribution():
  return TRAINING_DISTRIBUTION


DEFINITION = definitions.MultiSolAlgorithm(
    algorithm_name="mst_prim_multi",
    base_algorithm_name="mst_prim",
    spec=SPEC,
    sampler_class=samplers.MSTPrimMultiSampler,
    algorithm=generator.mst_prim_multi,
    evaluator=evaluate_mst_prim_multisol_batch,
    training_distribution=TRAINING_DISTRIBUTION,
    randomized_algorithm=RANDOMIZED_ALGORITHM,
    solution_space=SOLUTION_SPACE,
)
