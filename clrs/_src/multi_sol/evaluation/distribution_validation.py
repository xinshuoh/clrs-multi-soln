"""Distribution-validation helpers for multi-solution sampling plugins."""

from __future__ import annotations

import os
from typing import Callable, Dict, Iterable, Mapping, Tuple

from absl import logging
import numpy as np


def sample_n(
    sample_fn: Callable[[object], object],
    sample_input,
    n_samples: int,
    label: str | None = None,
) -> np.ndarray:
  """Draw `n_samples` batches of hard parent trees from a sampler."""
  samples = []
  log_every = max(1, n_samples // 5)
  if label:
    logging.info(
        'Distribution validation: sampling %s, 0/%d draws.',
        label,
        n_samples,
    )
  for sample_ix in range(n_samples):
    samples.append(np.asarray(sample_fn(sample_input)))
    if label and (
        sample_ix + 1 == n_samples or (sample_ix + 1) % log_every == 0):
      logging.info(
          'Distribution validation: sampling %s, %d/%d draws.',
          label,
          sample_ix + 1,
          n_samples,
      )
  return np.asarray(samples)


def summarize_samples(
    *,
    adjacency: np.ndarray,
    source_nodes: Iterable[int],
    samples: np.ndarray,
    validate_fn: Callable[[np.ndarray, object, int], bool],
    curve_max_graphs: int | None = None,
) -> Dict[str, object]:
  """Summarize validity and diversity for sampled parent trees."""
  adjacency = np.asarray(adjacency)
  source_nodes = np.asarray(source_nodes).astype(int)
  samples = np.asarray(samples)
  n_samples = int(samples.shape[0])

  unique_fractions = []
  valid_unique_fractions = []
  valid_fractions = []
  unique_counts = []
  valid_unique_counts = []
  valid_counts = []
  curve_rows = []

  for graph_ix in range(len(adjacency)):
    if graph_ix == 0:
      logging.info(
          'Distribution validation: summarizing %d graphs and %d samples.',
          len(adjacency),
          n_samples,
      )
    graph_samples = np.asarray(samples[:, graph_ix])
    unique_trees = _unique_parent_trees(graph_samples)
    unique_valids = [
        bool(validate_fn(adjacency[graph_ix], tree, int(source_nodes[graph_ix])))
        for tree in unique_trees
    ]
    sample_valids = [
        bool(validate_fn(adjacency[graph_ix], tree, int(source_nodes[graph_ix])))
        for tree in graph_samples
    ]

    unique_count = len(unique_trees)
    valid_unique_count = int(sum(unique_valids))
    valid_count = int(sum(sample_valids))
    unique_counts.append(unique_count)
    valid_unique_counts.append(valid_unique_count)
    valid_counts.append(valid_count)
    unique_fractions.append(unique_count / n_samples if n_samples else 0.0)
    valid_unique_fractions.append(
        valid_unique_count / unique_count if unique_count else 0.0)
    valid_fractions.append(valid_count / n_samples if n_samples else 0.0)
    if curve_max_graphs is None or graph_ix < curve_max_graphs:
      curve_rows.extend(_curve_rows_for_graph(
          graph_ix=graph_ix,
          adjacency=adjacency[graph_ix],
          source=int(source_nodes[graph_ix]),
          samples=graph_samples,
          sample_valids=sample_valids,
      ))

  return {
      "uniques": unique_fractions,
      "valids_uniques": valid_unique_fractions,
      "valids": valid_fractions,
      "unique_counts": unique_counts,
      "valid_unique_counts": valid_unique_counts,
      "valid_counts": valid_counts,
      "curves": curve_rows,
  }


def evaluate_sampling_methods(
    *,
    methods: Mapping[str, Callable[[object], object]],
    model_input,
    true_input,
    adjacency: np.ndarray,
    source_nodes: Iterable[int],
    validate_fn: Callable[[np.ndarray, object, int], bool],
    n_samples: int,
    curve_max_graphs: int | None = None,
) -> Dict[str, object]:
  """Evaluate all model/target sampling methods with shared metrics."""
  result_dict = {}
  scalar_metrics = {}
  curve_rows = []
  n_samples = max(1, int(n_samples))

  for method_name, sample_fn in methods.items():
    for value_name, sample_input in (("Model", model_input), ("True", true_input)):
      label = f"{method_name}/{value_name}"
      logging.info('Distribution validation: evaluating %s.', label)
      samples = sample_n(sample_fn, sample_input, n_samples, label=label)
      summary = summarize_samples(
          adjacency=adjacency,
          source_nodes=source_nodes,
          samples=samples,
          validate_fn=validate_fn,
          curve_max_graphs=curve_max_graphs,
      )
      prefix = f"{method_name}_{value_name}"
      result_dict[f"{prefix}_Uniques"] = summary["uniques"]
      result_dict[f"{prefix}_Valids_Uniques"] = summary["valids_uniques"]
      result_dict[f"{prefix}_Valids"] = summary["valids"]
      result_dict[f"{prefix}_Unique_Counts"] = summary["unique_counts"]
      result_dict[f"{prefix}_Valid_Unique_Counts"] = summary[
          "valid_unique_counts"]
      result_dict[f"{prefix}_Valid_Counts"] = summary["valid_counts"]
      scalar_metrics[f"{prefix}_Uniqueness"] = _mean(summary["uniques"])
      scalar_metrics[f"{prefix}_Valid_Unique"] = _mean(
          summary["valids_uniques"])
      scalar_metrics[f"{prefix}_Valid"] = _mean(summary["valids"])

      for row in summary["curves"]:
        row = dict(row)
        row["Method"] = method_name
        row["Source"] = value_name
        curve_rows.append(row)
      logging.info('Distribution validation: finished %s.', label)

  return {
      "result_dict": result_dict,
      "scalar_metrics": scalar_metrics,
      "curves": curve_rows,
  }


def evaluate_mixed_sampling_methods(
    *,
    model_methods: Mapping[str, Callable[[object], object]],
    true_methods: Mapping[str, Callable[[object], object]],
    model_input,
    true_input,
    adjacency: np.ndarray,
    source_nodes: Iterable[int],
    validate_fn: Callable[[np.ndarray, object, int], bool],
    n_samples: int,
    curve_max_graphs: int | None = None,
) -> Dict[str, object]:
  """Evaluate methods whose model/target call signatures differ."""
  result_dict = {}
  scalar_metrics = {}
  curve_rows = []
  n_samples = max(1, int(n_samples))

  for method_name, model_fn in model_methods.items():
    for value_name, sample_fn, sample_input in (
        ("Model", model_fn, model_input),
        ("True", true_methods[method_name], true_input),
    ):
      label = f"{method_name}/{value_name}"
      logging.info('Distribution validation: evaluating %s.', label)
      samples = sample_n(sample_fn, sample_input, n_samples, label=label)
      summary = summarize_samples(
          adjacency=adjacency,
          source_nodes=source_nodes,
          samples=samples,
          validate_fn=validate_fn,
          curve_max_graphs=curve_max_graphs,
      )
      prefix = f"{method_name}_{value_name}"
      result_dict[f"{prefix}_Uniques"] = summary["uniques"]
      result_dict[f"{prefix}_Valids_Uniques"] = summary["valids_uniques"]
      result_dict[f"{prefix}_Valids"] = summary["valids"]
      result_dict[f"{prefix}_Unique_Counts"] = summary["unique_counts"]
      result_dict[f"{prefix}_Valid_Unique_Counts"] = summary[
          "valid_unique_counts"]
      result_dict[f"{prefix}_Valid_Counts"] = summary["valid_counts"]
      scalar_metrics[f"{prefix}_Uniqueness"] = _mean(summary["uniques"])
      scalar_metrics[f"{prefix}_Valid_Unique"] = _mean(
          summary["valids_uniques"])
      scalar_metrics[f"{prefix}_Valid"] = _mean(summary["valids"])

      for row in summary["curves"]:
        row = dict(row)
        row["Method"] = method_name
        row["Source"] = value_name
        curve_rows.append(row)
      logging.info('Distribution validation: finished %s.', label)

  return {
      "result_dict": result_dict,
      "scalar_metrics": scalar_metrics,
      "curves": curve_rows,
  }


def evaluate_sampling_sources(
    *,
    sources: Mapping[Tuple[str, str], Tuple[Callable[[object], object], object]],
    adjacency: np.ndarray,
    source_nodes: Iterable[int],
    validate_fn: Callable[[np.ndarray, object, int], bool],
    n_samples: int,
    curve_max_graphs: int | None = None,
) -> Dict[str, object]:
  """Evaluate arbitrary sampling sources with the shared curve machinery.

  `sources` maps `(method_name, source_name)` to `(sample_fn, sample_input)`.
  This is intended for Appendix-C style comparators such as repeated symbolic
  randomized algorithm runs, which do not naturally fit the Model/True pairing.
  """
  result_dict = {}
  scalar_metrics = {}
  curve_rows = []
  n_samples = max(1, int(n_samples))

  for (method_name, source_name), (sample_fn, sample_input) in sources.items():
    label = f"{method_name}/{source_name}"
    logging.info('Distribution validation: evaluating %s.', label)
    samples = sample_n(sample_fn, sample_input, n_samples, label=label)
    summary = summarize_samples(
        adjacency=adjacency,
        source_nodes=source_nodes,
        samples=samples,
        validate_fn=validate_fn,
        curve_max_graphs=curve_max_graphs,
    )
    prefix = f"{method_name}_{source_name}"
    result_dict[f"{prefix}_Uniques"] = summary["uniques"]
    result_dict[f"{prefix}_Valids_Uniques"] = summary["valids_uniques"]
    result_dict[f"{prefix}_Valids"] = summary["valids"]
    result_dict[f"{prefix}_Unique_Counts"] = summary["unique_counts"]
    result_dict[f"{prefix}_Valid_Unique_Counts"] = summary[
        "valid_unique_counts"]
    result_dict[f"{prefix}_Valid_Counts"] = summary["valid_counts"]
    scalar_metrics[f"{prefix}_Uniqueness"] = _mean(summary["uniques"])
    scalar_metrics[f"{prefix}_Valid_Unique"] = _mean(
        summary["valids_uniques"])
    scalar_metrics[f"{prefix}_Valid"] = _mean(summary["valids"])

    for row in summary["curves"]:
      row = dict(row)
      row["Method"] = method_name
      row["Source"] = source_name
      curve_rows.append(row)
    logging.info('Distribution validation: finished %s.', label)

  return {
      "result_dict": result_dict,
      "scalar_metrics": scalar_metrics,
      "curves": curve_rows,
  }


def save_sampling_curve_artifacts(
    curve_rows,
    *,
    filename: str,
    output_dir: str = ".",
) -> None:
  """Persist uniqueness and edge-reuse curve data.

  CSVs are the default artifact because they are stable on headless compute
  nodes. Set `CLRS_MULTISOL_SAVE_PLOTS=1` to also emit PNG plots.
  """
  if not curve_rows:
    return

  os.makedirs(output_dir, exist_ok=True)
  import pandas as pd  # Lazy import; only needed for optional artifacts.

  df = pd.DataFrame.from_records(curve_rows)
  csv_path = os.path.join(output_dir, f"{filename}_curves.csv")
  df.to_csv(csv_path, index=False)
  summary_path = os.path.join(output_dir, f"{filename}_curve_summary.csv")
  summary_df = (
      df.groupby(["Method", "Source", "Samples"])[
          ["Cumulative_Valid_Unique", "Edge_Reuse_Mean", "Edge_Reuse_Median"]
      ]
      .agg(["mean", "std"])
      .reset_index()
  )
  summary_df.columns = [
      "_".join(str(part) for part in col if part)
      if isinstance(col, tuple)
      else col
      for col in summary_df.columns
  ]
  summary_df.to_csv(summary_path, index=False)

  if os.environ.get("CLRS_MULTISOL_SAVE_PLOTS") == "1":
    _plot_curve(
        df,
        value_col="Cumulative_Valid_Unique",
        ylabel="Unique and valid solutions",
        path=os.path.join(output_dir, f"{filename}_unique_valid_curve.png"),
    )
    _plot_curve(
        df,
        value_col="Edge_Reuse_Mean",
        ylabel="Mean edge reuse",
        path=os.path.join(output_dir, f"{filename}_edge_reuse_curve.png"),
    )


def edge_reuse_stats(parent_trees: np.ndarray, num_nodes: int) -> Dict[str, float]:
  """Return cumulative nonzero edge-reuse mean/median for sampled trees."""
  matrices = [
      parent_tree_to_edge_matrix(tree, num_nodes)
      for tree in np.asarray(parent_trees)
  ]
  if not matrices:
    return {"mean": 0.0, "median": 0.0}
  edge_counts = np.sum(matrices, axis=0)
  edge_freqs = edge_counts[edge_counts > 0] / float(len(matrices))
  if edge_freqs.size == 0:
    return {"mean": 0.0, "median": 0.0}
  return {
      "mean": float(np.mean(edge_freqs)),
      "median": float(np.median(edge_freqs)),
  }


def parent_tree_to_edge_matrix(parent_tree, num_nodes: int) -> np.ndarray:
  """Convert a parent tree to a directed adjacency matrix, excluding self-loops."""
  matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
  for child, parent in enumerate(np.asarray(parent_tree).astype(int)):
    if parent < 0 or parent >= num_nodes or parent == child:
      continue
    matrix[parent, child] = 1.0
  return matrix


def _unique_parent_trees(trees: np.ndarray):
  return [np.asarray(item) for item in {
      tuple(np.asarray(row).astype(int).tolist()) for row in trees
  }]


def _curve_rows_for_graph(
    *,
    graph_ix: int,
    adjacency: np.ndarray,
    source: int,
    samples: np.ndarray,
    sample_valids,
):
  seen = set()
  seen_valid = set()
  valid_count = 0
  rows = []
  for sample_ix, tree in enumerate(samples, start=1):
    key = tuple(np.asarray(tree).astype(int).tolist())
    sample_unique = key not in seen
    seen.add(key)
    sample_valid = bool(sample_valids[sample_ix - 1])
    sample_valid_unique = sample_valid and key not in seen_valid
    if sample_valid:
      valid_count += 1
      seen_valid.add(key)
    reuse = edge_reuse_stats(samples[:sample_ix], adjacency.shape[0])
    rows.append({
        "Graph": graph_ix,
        "Source_Node": source,
        "Samples": sample_ix,
        "Sample_Valid": sample_valid,
        "Sample_Unique": sample_unique,
        "Sample_Valid_Unique": sample_valid_unique,
        "Solution_Key": "|".join(str(item) for item in key),
        "Parent_Tree": list(key),
        "Cumulative_Unique": len(seen),
        "Cumulative_Valid": valid_count,
        "Cumulative_Valid_Unique": len(seen_valid),
        "Edge_Reuse_Mean": reuse["mean"],
        "Edge_Reuse_Median": reuse["median"],
    })
  return rows


def _plot_curve(df, *, value_col: str, ylabel: str, path: str) -> None:
  import matplotlib.pyplot as plt  # Lazy import; only needed for artifacts.

  grouped = (
      df.groupby(["Method", "Source", "Samples"])[value_col]
      .agg(["mean", "std"])
      .reset_index()
  )
  plt.figure()
  for (method, source), subdf in grouped.groupby(["Method", "Source"]):
    x = subdf["Samples"].to_numpy()
    mean = subdf["mean"].to_numpy()
    std = np.nan_to_num(subdf["std"].to_numpy())
    label = f"{method} {source}"
    plt.plot(x, mean, marker="o", linewidth=1.5, markersize=3, label=label)
    plt.fill_between(x, mean - std, mean + std, alpha=0.15)
  plt.xlabel("Sampled solutions")
  plt.ylabel(ylabel)
  plt.legend(loc="best", fontsize="small")
  plt.tight_layout()
  plt.savefig(path)
  plt.close()


def _mean(values) -> float:
  return float(np.mean(values)) if len(values) else 0.0


def run_bf_distribution_validation(
    *,
    adjacency,
    source_nodes,
    outputs,
    preds,
    nse: int,
    output_dir: str = ".",
) -> None:
  """Run Bellman-Ford legacy distribution-validation side effects."""
  from clrs._src.multi_sol.evaluation.distribution_generation import (
      build_bf_validation_payload,
      generate_validation_dataframes,
  )
  from clrs._src.validate_distributions import (
      line_plot,
      plot_edge_reuse_matrix_list_mean,
      plot_n_unique_by_n_extracted,
  )

  payload = build_bf_validation_payload(
      adjacency=adjacency,
      source_nodes=source_nodes,
      outputs=outputs,
      preds=preds,
  )
  uniqueness_dataframes, edge_reuse_df = generate_validation_dataframes(
      payload=payload,
      nse=nse,
      mode="BF",
  )
  plot_n_unique_by_n_extracted(
      uniqueness_dataframes, payload.graph_size, output_dir=output_dir)
  plot_edge_reuse_matrix_list_mean(
      edge_reuse_df, payload.graph_size, output_dir=output_dir)
  line_plot(edge_reuse_df, payload.graph_size, output_dir=output_dir)


def run_dfs_distribution_validation(
    *,
    adjacency,
    outputs,
    pred_batches,
    nse: int,
    output_dir: str = ".",
) -> None:
  """Run DFS legacy distribution-validation side effects."""
  from clrs._src.multi_sol.evaluation.distribution_generation import (
      build_dfs_validation_payload,
      generate_validation_dataframes,
  )
  from clrs._src.validate_distributions import (
      line_plot_dfs,
      plot_edge_reuse_matrix_list_mean_dfs,
      plot_n_unique_by_n_extracted_dfs,
  )

  payload = build_dfs_validation_payload(
      adjacency=adjacency,
      outputs=outputs,
      preds=pred_batches,
  )
  uniqueness_dataframes, edge_reuse_df = generate_validation_dataframes(
      payload=payload,
      nse=nse,
      mode="DFS",
  )
  plot_n_unique_by_n_extracted_dfs(
      uniqueness_dataframes, payload.graph_size, output_dir=output_dir)
  plot_edge_reuse_matrix_list_mean_dfs(
      edge_reuse_df, payload.graph_size, output_dir=output_dir)
  line_plot_dfs(edge_reuse_df, payload.graph_size, output_dir=output_dir)
