"""Declarative extension definition for ``dfs_multi``."""

from typing import Dict

from clrs._src import specs
from clrs._src.multi_sol.algorithms.dfs.plugin import evaluate_dfs_multisol_batch
from clrs._src.multi_sol.core.registry import MultiSolAlgorithmExtension


def _dfs_multi_spec_factory(_base_specs_map: Dict[str, specs.Spec]) -> specs.Spec:
  return {
      'pos': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.SCALAR),
      'A': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.SCALAR),
      'adj': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
      'pi': (
          specs.Stage.OUTPUT,
          specs.Location.NODE,
          specs.Type.POINTER_DISTRIBUTION,
      ),
      'pi_h': (specs.Stage.HINT, specs.Location.NODE, specs.Type.POINTER),
      'color': (specs.Stage.HINT, specs.Location.NODE, specs.Type.CATEGORICAL),
      'd': (specs.Stage.HINT, specs.Location.NODE, specs.Type.SCALAR),
      'f': (specs.Stage.HINT, specs.Location.NODE, specs.Type.SCALAR),
      's_prev': (specs.Stage.HINT, specs.Location.NODE, specs.Type.POINTER),
      's': (specs.Stage.HINT, specs.Location.NODE, specs.Type.MASK_ONE),
      'u': (specs.Stage.HINT, specs.Location.NODE, specs.Type.MASK_ONE),
      'v': (specs.Stage.HINT, specs.Location.NODE, specs.Type.MASK_ONE),
      's_last': (specs.Stage.HINT, specs.Location.NODE, specs.Type.MASK_ONE),
      'time': (specs.Stage.HINT, specs.Location.GRAPH, specs.Type.SCALAR),
  }


def _dfs_multi_sampler_factory():
  from clrs._src.multi_sol import samplers as sampler_module
  return sampler_module.DfsMultiSampler


def _dfs_multi_algorithm_factory():
  from clrs._src.multi_sol.algorithms import multi_graphs
  return multi_graphs.dfs_multi


EXTENSION = MultiSolAlgorithmExtension(
    algorithm_name="dfs_multi",
    base_algorithm_name="dfs",
    output_name="pi",
    spec_factory=_dfs_multi_spec_factory,
    sampler_factory=_dfs_multi_sampler_factory,
    algorithm_factory=_dfs_multi_algorithm_factory,
    evaluator=evaluate_dfs_multisol_batch,
)

__all__ = ("EXTENSION",)
