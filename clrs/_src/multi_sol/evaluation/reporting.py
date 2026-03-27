"""Artifact/report persistence sinks for multi-solution evaluation."""

from __future__ import annotations

from datetime import datetime
import os
import pickle
from typing import Any, Dict


def discard_report(_result_dict: Dict[str, Any], _filename: str) -> None:
  """No-op sink for side-effect free evaluation."""


def save_pickle_report(result_dict: Dict[str, Any], filename: str) -> None:
  """Persist report payload to `results/` as timestamped pickle."""
  os.makedirs("results", exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  save_path = os.path.join("results", f"{filename}_{timestamp}.pkl")
  with open(save_path, "wb") as f:
    pickle.dump(result_dict, f)


def save_csv_report(result_dict: Dict[str, Any], filename: str) -> None:
  """Persist report payload to `results/` as timestamped CSV."""
  import pandas as pd  # Lazy import; needed only for CSV sinks.

  os.makedirs("results", exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  save_path = os.path.join("results", f"{filename}_{timestamp}.csv")
  pd.DataFrame.from_dict(result_dict).to_csv(
      save_path, encoding="utf-8", index=False)
  print(f"Saved results to {save_path}")
