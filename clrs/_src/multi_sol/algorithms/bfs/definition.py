"""Declarative extension definition for ``bfs_multi``."""

from typing import Dict

from clrs._src import specs
from clrs._src.multi_sol.algorithms.bfs.plugin import evaluate_bfs_multisol_batch
from clrs._src.multi_sol.core.registry import MultiSolAlgorithmExtension


def _bfs_multi_spec_factory(_base_specs_map: Dict[str, specs.Spec]) -> specs.Spec:
  return {
      'pos': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.SCALAR),
      's': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.MASK_ONE),
      'A': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.SCALAR),
      'adj': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
      'pi': (
          specs.Stage.OUTPUT,
          specs.Location.NODE,
          specs.Type.POINTER_DISTRIBUTION,
      ),
      'reach_h': (specs.Stage.HINT, specs.Location.NODE, specs.Type.MASK),
      'pi_h': (specs.Stage.HINT, specs.Location.NODE, specs.Type.POINTER),
  }


def _bfs_multi_sampler_factory():
  from clrs._src.multi_sol import samplers as sampler_module
  return sampler_module.BfsMultiSampler


def _bfs_multi_algorithm_factory():
  from clrs._src.multi_sol.algorithms import multi_graphs
  return multi_graphs.bfs_multi


EXTENSION = MultiSolAlgorithmExtension(
    algorithm_name="bfs_multi",
    base_algorithm_name="bfs",
    output_name="pi",
    spec_factory=_bfs_multi_spec_factory,
    sampler_factory=_bfs_multi_sampler_factory,
    algorithm_factory=_bfs_multi_algorithm_factory,
    evaluator=evaluate_bfs_multisol_batch,
)

__all__ = ("EXTENSION",)
