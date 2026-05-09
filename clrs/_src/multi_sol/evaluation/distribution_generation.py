"""Reusable orchestration for distribution-validation dataframe generation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import numpy as np

from clrs._src.validate_distributions import validate_distributions


@dataclass(frozen=True)
class DistributionValidationPayload:
  """Input bundle for legacy distribution-validation generation."""

  adjacency: np.ndarray
  source_nodes: object
  outs_or_preds: object
  graph_size: int


def build_bf_validation_payload(
    adjacency: np.ndarray,
    source_nodes: np.ndarray,
    outputs,
    preds,
    max_graphs: int = 10,
) -> DistributionValidationPayload:
  """Build Bellman-Ford payload preserving legacy truncation semantics."""
  adjacency_vd = adjacency[:max_graphs]
  source_nodes_vd = source_nodes[:max_graphs]
  outputs_vd = copy.deepcopy(outputs)
  preds_vd = copy.deepcopy(preds)
  outputs_vd[0].data = outputs_vd[0].data[:max_graphs]
  preds_vd["pi"].data = preds_vd["pi"].data[:max_graphs]
  return DistributionValidationPayload(
      adjacency=adjacency_vd,
      source_nodes=source_nodes_vd,
      outs_or_preds=[preds_vd],
      graph_size=len(adjacency[0]),
  )


def build_dfs_validation_payload(
    adjacency: np.ndarray,
    outputs,
    preds,
    max_graphs: int = 10,
) -> DistributionValidationPayload:
  """Build DFS payload preserving legacy truncation semantics."""
  adjacency_vd = adjacency[:max_graphs]
  outputs_vd = copy.deepcopy(outputs)
  preds_vd = copy.deepcopy(preds)
  outputs_vd[0].data = outputs_vd[0].data[:max_graphs]
  preds_vd[0]["pi"].data = preds_vd[0]["pi"].data[:max_graphs]
  return DistributionValidationPayload(
      adjacency=adjacency_vd,
      source_nodes=[0] * len(adjacency_vd),
      outs_or_preds=preds_vd,
      graph_size=len(adjacency[0]),
  )


def generate_validation_dataframes(
    payload: DistributionValidationPayload,
    nse: int,
    mode: Literal["BF", "DFS"],
):
  """Generate uniqueness and edge-reuse dataframe lists via legacy validators."""
  if mode == "BF":
    uniqueness_dataframes, _, _ = validate_distributions(
        As=payload.adjacency,
        Ss=payload.source_nodes,
        outsOrPreds=payload.outs_or_preds,
        numSolsExtracting=nse,
        flag="BF",
    )
    edge_reuse_dataframes, _, _ = validate_distributions(
        As=payload.adjacency,
        Ss=payload.source_nodes,
        outsOrPreds=payload.outs_or_preds,
        numSolsExtracting=nse,
        flag="dummy",
        edge_reuse_BF=True,
    )
    return uniqueness_dataframes, edge_reuse_dataframes

  uniqueness_dataframes, _, _ = validate_distributions(
      As=payload.adjacency,
      Ss=payload.source_nodes,
      outsOrPreds=payload.outs_or_preds,
      numSolsExtracting=nse,
      flag="DFS",
  )
  edge_reuse_dataframes, _, _ = validate_distributions(
      As=payload.adjacency,
      Ss=payload.source_nodes,
      outsOrPreds=payload.outs_or_preds,
      numSolsExtracting=nse,
      flag="dummy",
      edge_reuse_DFS=True,
  )
  return uniqueness_dataframes, edge_reuse_dataframes
