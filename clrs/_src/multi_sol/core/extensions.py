"""Built-in multi-solution algorithm extension registrations.

Use `specs.Type.MULTI_SOLUTION` for extension wiring.
`specs.Type.MULT_SOL` remains a temporary compatibility alias.
"""

from __future__ import annotations

from typing import Callable

from clrs._src import specs
from clrs._src.multi_sol.core import registry


def _pointer_to_multisol_transform(output_name: str) -> Callable[[specs.Spec], specs.Spec]:
  def _transform(base_spec: specs.Spec) -> specs.Spec:
    cloned = dict(base_spec)
    stage, loc, _ = cloned[output_name]
    cloned[output_name] = (stage, loc, specs.Type.MULTI_SOLUTION)
    return cloned

  return _transform


def _multisol_sampler_factory(sampler_name: str):
  def _factory():
    from clrs._src import samplers as sampler_module
    if sampler_name == "dfs_multi":
      return sampler_module.DfsMultiSampler
    if sampler_name == "bfs_multi":
      return sampler_module.BfsMultiSampler
    if sampler_name == "bellman_ford_multi":
      return sampler_module.BellmanFordMultiSampler
    raise KeyError(f"Unknown multi-solution sampler {sampler_name}.")

  return _factory


def _evaluate_dfs(**kwargs):
  from clrs._src.multi_sol.algorithms.dfs.plugin import evaluate_dfs_multisol_batch
  return evaluate_dfs_multisol_batch(**kwargs)


def _evaluate_bfs(**kwargs):
  from clrs._src.multi_sol.algorithms.bfs.plugin import evaluate_bfs_multisol_batch
  return evaluate_bfs_multisol_batch(**kwargs)


def _evaluate_bf(**kwargs):
  from clrs._src.multi_sol.algorithms.bellman_ford.plugin import evaluate_bf_multisol_batch
  return evaluate_bf_multisol_batch(**kwargs)


def _register_builtin_extensions() -> None:
  if registry.get_extension("dfs_multi") is not None:
    return

  registry.register_extension(
      registry.MultiSolAlgorithmExtension(
          algorithm_name="dfs_multi",
          base_algorithm_name="dfs",
          output_name="pi",
          spec_transformer=_pointer_to_multisol_transform("pi"),
          sampler_factory=_multisol_sampler_factory("dfs_multi"),
          evaluator=_evaluate_dfs,
      ))

  registry.register_extension(
      registry.MultiSolAlgorithmExtension(
          algorithm_name="bfs_multi",
          base_algorithm_name="bfs",
          output_name="pi",
          spec_transformer=_pointer_to_multisol_transform("pi"),
          sampler_factory=_multisol_sampler_factory("bfs_multi"),
          evaluator=_evaluate_bfs,
      ))

  registry.register_extension(
      registry.MultiSolAlgorithmExtension(
          algorithm_name="bellman_ford_multi",
          base_algorithm_name="bellman_ford",
          output_name="pi",
          spec_transformer=_pointer_to_multisol_transform("pi"),
          sampler_factory=_multisol_sampler_factory("bellman_ford_multi"),
          evaluator=_evaluate_bf,
      ))


_register_builtin_extensions()
