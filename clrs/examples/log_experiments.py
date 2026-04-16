"""Deprecated compatibility wrappers for legacy multi-solution entrypoints.

This module is kept only for backward compatibility with callers that still
import its helper functions. The canonical execution path is
`python -m clrs.examples.run`.
"""

import warnings

from clrs._src.multi_sol.algorithms.bellman_ford.plugin import (
    evaluate_bf_multisol_batch,
)
from clrs._src.multi_sol.algorithms.bfs.plugin import evaluate_bfs_multisol_batch
from clrs._src.multi_sol.algorithms.dfs.plugin import evaluate_dfs_multisol_batch
from clrs._src.multi_sol.evaluation import reporting

_DEPRECATION_MESSAGE = (
    "`clrs.examples.log_experiments` is deprecated and compatibility-only. "
    "Use `python -m clrs.examples.run` for training/evaluation orchestration."
)


def _warn_deprecated(symbol):
  warnings.warn(
      f"{_DEPRECATION_MESSAGE} Called `{symbol}`.",
      category=DeprecationWarning,
      stacklevel=2,
  )


def save_results(result_dict, filename):
  """Compatibility alias for CSV report sink."""
  _warn_deprecated("save_results")
  reporting.save_csv_report(
      result_dict,
      filename,
      output_dir='.',
      timestamped=False,
  )


def BF_collect_and_eval(
    sampler,
    predict_fn,
    sample_count,
    rng_key,
    extras,
    filename="bf_accuracy",
    vd_flag=True,
    NSE=100,
    save_results_fn=None,
):
  """Compatibility wrapper for Bellman-Ford multi-solution evaluation."""
  _warn_deprecated("BF_collect_and_eval")
  report_sink = save_results_fn or save_results
  return evaluate_bf_multisol_batch(
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      extras=extras,
      save_results_fn=report_sink,
      filename=filename,
      vd_flag=vd_flag,
      NSE=NSE,
  )


def BFS_multi_collect_and_eval(
    sampler,
    predict_fn,
    sample_count,
    rng_key,
    extras,
    filename="bfs_accuracy",
    vd_flag=False,
    NSE=100,
    save_results_fn=None,
):
  """Compatibility wrapper for BFS multi-solution evaluation."""
  _warn_deprecated("BFS_multi_collect_and_eval")
  del vd_flag, NSE  # kept for CLI compatibility
  report_sink = save_results_fn or save_results
  return evaluate_bfs_multisol_batch(
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      extras=extras,
      save_results_fn=report_sink,
      filename=filename,
  )


def DFS_collect_and_eval(
    sampler,
    predict_fn,
    sample_count,
    rng_key,
    extras,
    filename="dfs_accuracy",
    vd_flag=False,
    NSE=100,
    save_results_fn=None,
):
  """Compatibility wrapper for DFS multi-solution evaluation."""
  _warn_deprecated("DFS_collect_and_eval")
  report_sink = save_results_fn or save_results
  return evaluate_dfs_multisol_batch(
      sampler=sampler,
      predict_fn=predict_fn,
      sample_count=sample_count,
      rng_key=rng_key,
      extras=extras,
      save_results_fn=report_sink,
      filename=filename,
      vd_flag=vd_flag,
      NSE=NSE,
  )


def main():
  """Deprecated CLI shim; use clrs.examples.run instead."""
  _warn_deprecated("__main__")
  raise SystemExit(
      "Deprecated compatibility-only module. "
      "Run `python -m clrs.examples.run` instead."
  )


if __name__ == "__main__":
  main()
