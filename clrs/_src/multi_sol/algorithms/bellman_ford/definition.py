"""Declarative extension definition for ``bellman_ford_multi``."""

from typing import Dict

from clrs._src import specs
from clrs._src.multi_sol.algorithms.bellman_ford.plugin import evaluate_bf_multisol_batch
from clrs._src.multi_sol.core.registry import MultiSolAlgorithmExtension


def _bellman_ford_multi_spec_factory(
    _base_specs_map: Dict[str, specs.Spec]) -> specs.Spec:
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
      'pi_h': (specs.Stage.HINT, specs.Location.NODE, specs.Type.POINTER),
      'd': (specs.Stage.HINT, specs.Location.NODE, specs.Type.SCALAR),
      'msk': (specs.Stage.HINT, specs.Location.NODE, specs.Type.MASK),
  }


def _bellman_ford_multi_sampler_factory():
  from clrs._src.multi_sol import samplers as sampler_module
  return sampler_module.BellmanFordMultiSampler


def _bellman_ford_multi_algorithm_factory():
  from clrs._src.multi_sol.algorithms import multi_graphs
  return multi_graphs.bellman_ford_multi


EXTENSION = MultiSolAlgorithmExtension(
    algorithm_name="bellman_ford_multi",
    base_algorithm_name="bellman_ford",
    output_name="pi",
    spec_factory=_bellman_ford_multi_spec_factory,
    sampler_factory=_bellman_ford_multi_sampler_factory,
    algorithm_factory=_bellman_ford_multi_algorithm_factory,
    evaluator=evaluate_bf_multisol_batch,
)

__all__ = ("EXTENSION",)
