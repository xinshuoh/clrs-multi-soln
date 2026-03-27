"""Optional distribution-validation orchestration for DFS/Bellman-Ford plugins."""

from __future__ import annotations


def run_bf_distribution_validation(
    *,
    adjacency,
    source_nodes,
    outputs,
    preds,
    nse: int,
) -> None:
  """Run Bellman-Ford legacy distribution-validation side effects."""
  from clrs._src.multi_sol.data.distribution_generation import (
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
  plot_n_unique_by_n_extracted(uniqueness_dataframes, payload.graph_size)
  plot_edge_reuse_matrix_list_mean(edge_reuse_df, payload.graph_size)
  line_plot(edge_reuse_df, payload.graph_size)


def run_dfs_distribution_validation(
    *,
    adjacency,
    outputs,
    pred_batches,
    nse: int,
) -> None:
  """Run DFS legacy distribution-validation side effects."""
  from clrs._src.multi_sol.data.distribution_generation import (
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
  plot_n_unique_by_n_extracted_dfs(uniqueness_dataframes, payload.graph_size)
  plot_edge_reuse_matrix_list_mean_dfs(edge_reuse_df, payload.graph_size)
  line_plot_dfs(edge_reuse_df, payload.graph_size)
