"""Legacy BFS multi-solution evaluation entry points."""

from typing import Dict

from clrs._src.multi_sol.algorithms import graphs
from clrs._src.multi_sol.algorithms.bfs import definition
from clrs._src.multi_sol.evaluation import batch_evaluation


distribution_validation = batch_evaluation.distribution_validation
_sample_randomized_bfs_algorithm = definition._sample_randomized_bfs_algorithm
_randomized_bfs_tree = graphs.bfs_multi


def evaluate_bfs_multisol_batch(
    sampler,
    predict_fn,
    sample_count,
    rng_key,
    extras,
    save_results_fn=None,
    filename="bfs_accuracy",
    vd_flag=False,
    NSE=100,
    output_dir=".",
    curve_max_graphs=None,
) -> Dict[str, float]:
  """Compatibility wrapper around the definition-driven evaluator."""
  return definition.evaluate_bfs_multisol_batch(
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      extras=extras,
      save_results_fn=save_results_fn,
      filename=filename,
      vd_flag=vd_flag,
      NSE=NSE,
      output_dir=output_dir,
      curve_max_graphs=curve_max_graphs,
  )
