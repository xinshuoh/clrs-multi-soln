"""Artifact/report persistence sinks for multi-solution evaluation."""

from __future__ import annotations

from datetime import datetime
import os
import pickle
from typing import Any, Dict


def discard_report(_result_dict: Dict[str, Any], _filename: str) -> None:
  """No-op sink for side-effect free evaluation."""
  return None


def _resolve_artifact_path(
    filename: str,
    *,
    extension: str,
    output_dir: str = ".",
    timestamped: bool = False,
) -> str:
  os.makedirs(output_dir, exist_ok=True)
  stem = filename
  if timestamped:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{filename}_{timestamp}"
  if not extension.startswith("."):
    extension = f".{extension}"
  if stem.endswith(extension):
    output_name = stem
  else:
    output_name = f"{stem}{extension}"
  return os.path.join(output_dir, output_name)


def save_pickle_report(
    result_dict: Dict[str, Any],
    filename: str,
    *,
    output_dir: str = ".",
    timestamped: bool = False,
) -> str:
  """Persist report payload as pickle."""
  output_path = _resolve_artifact_path(
      filename,
      extension=".pkl",
      output_dir=output_dir,
      timestamped=timestamped,
  )
  with open(output_path, "wb") as f:
    pickle.dump(result_dict, f)
  return output_path


def save_csv_report(
    rows,
    filename: str,
    *,
    output_dir: str = ".",
    timestamped: bool = False,
) -> str:
  """Persist row dictionaries to CSV."""
  import pandas as pd  # Lazy import; needed only for CSV sinks.

  output_path = _resolve_artifact_path(
      filename,
      extension=".csv",
      output_dir=output_dir,
      timestamped=timestamped,
  )
  pd.DataFrame(rows).to_csv(output_path, index=False)
  return output_path
