"""Artifact/report persistence sinks for multi-solution evaluation."""

from __future__ import annotations

from datetime import datetime
import os
import pickle
from typing import Any, Dict


def discard_report(_result_dict: Dict[str, Any], _filename: str) -> None:
  """No-op sink for side-effect free evaluation."""


def _resolve_artifact_path(
    *,
    filename: str,
    extension: str,
    output_dir: str,
    timestamped: bool,
) -> str:
  base_name = filename if filename.endswith(extension) else f"{filename}{extension}"
  if timestamped:
    stem, ext = os.path.splitext(base_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{stem}_{timestamp}{ext}"

  save_path = os.path.join(output_dir, base_name) if output_dir else base_name
  save_dir = os.path.dirname(save_path)
  if save_dir:
    os.makedirs(save_dir, exist_ok=True)
  return save_path


def save_pickle_report(
    result_dict: Dict[str, Any],
    filename: str,
    *,
    output_dir: str = "results",
    timestamped: bool = True,
) -> None:
  """Persist report payload as pickle."""
  save_path = _resolve_artifact_path(
      filename=filename,
      extension=".pkl",
      output_dir=output_dir,
      timestamped=timestamped,
  )
  with open(save_path, "wb") as f:
    pickle.dump(result_dict, f)


def save_csv_report(
    result_dict: Dict[str, Any],
    filename: str,
    *,
    output_dir: str = "results",
    timestamped: bool = True,
) -> None:
  """Persist report payload as CSV."""
  import pandas as pd  # Lazy import; needed only for CSV sinks.

  save_path = _resolve_artifact_path(
      filename=filename,
      extension=".csv",
      output_dir=output_dir,
      timestamped=timestamped,
  )
  pd.DataFrame.from_dict(result_dict).to_csv(
      save_path, encoding="utf-8", index=False)
  print(f"Saved results to {save_path}")
