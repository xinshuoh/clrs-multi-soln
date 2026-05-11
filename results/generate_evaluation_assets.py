"""Generate dissertation evaluation tables and figures from final result CSVs.

This script is intentionally reusable: it centralises the aggregation logic used
for the dissertation evaluation chapter instead of relying on ad hoc notebooks or
one-off shell snippets.

Usage:
  python results/generate_evaluation_assets.py
  python results/generate_evaluation_assets.py --root results/final
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SIZES = (5, 16, 64)
SOURCES = ("True", "Model")
PROCESSORS = ("triplet-gmpnn", "pgn")
DISPLAY_NAMES = {
    "altUpwards": "AltUpwards",
    "Prim": "Prim-like",
}
ALGORITHM_METHODS = {
    "bfs_multi": ("Categorical", "Prim", "Beam", "Random"),
    "dfs_multi": ("Argmax", "altUpwards", "Upwards", "Random"),
    "bellman_ford_multi": ("Argmax", "Beam", "Greedy", "Random"),
    "mst_prim_multi": ("Argmax", "Tree", "Greedy", "Random"),
}
ALGORITHM_DISPLAY_NAMES = {
    "bfs_multi": "BFS",
    "dfs_multi": "DFS",
    "bellman_ford_multi": "Bellman-Ford",
    "mst_prim_multi": "MST-Prim",
}
ALGORITHM_OUTPUT_NAMES = {
    "bfs_multi": "bfs",
    "dfs_multi": "dfs",
    "bellman_ford_multi": "bellman-ford",
    "mst_prim_multi": "mst-prim",
}


def _configure_matplotlib(output_dir: Path):
  """Import and configure matplotlib with a writable cache directory."""
  cache_dir = output_dir / ".matplotlib-cache"
  cache_dir.mkdir(parents=True, exist_ok=True)
  os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

  plt.rcParams.update({
      "font.size": 9,
      "axes.titlesize": 10,
      "axes.labelsize": 9,
      "legend.fontsize": 8,
      "figure.dpi": 160,
  })
  return plt


def score_files(
    root: Path,
    algorithm: str,
    processor: str = "triplet-gmpnn",
) -> list[Path]:
  """Return per-seed training score-result files for an algorithm."""
  pattern = root / algorithm / processor / "models" / "seed_*" / "score-results.csv"
  return [Path(p) for p in sorted(glob.glob(str(pattern)))]


def sampling_files(
    root: Path,
    algorithm: str,
    size: int,
    processor: str = "triplet-gmpnn",
) -> list[Path]:
  """Return per-seed n100 sampling report files for an algorithm and size."""
  pattern = (root / algorithm / processor / "eval" / f"test_length_{size}_n100" / "seed_*" /
             "samples.csv")
  return [Path(p) for p in sorted(glob.glob(str(pattern)))]


def load_score_frame(path: Path) -> pd.DataFrame:
  """Load a CLRS score-results CSV with normalised column names."""
  frame = pd.read_csv(path)
  return frame.rename(
      columns={
          "Num Steps": "step",
          "Train KlDiv": "kl",
          "Mean 1-abs(error)": "soft_accuracy",
          "Examples Seen": "examples_seen",
      })


def aggregate_learning_curve(
    root: Path,
    algorithm: str,
    processor: str = "triplet-gmpnn",
) -> pd.DataFrame:
  """Aggregate KL and soft-accuracy curves across seeds."""
  frames = []
  for seed_idx, path in enumerate(score_files(root, algorithm, processor)):
    frame = load_score_frame(path)
    frame = frame[["step", "kl", "soft_accuracy"]].copy()
    frame["seed_idx"] = seed_idx
    frames.append(frame)
  if not frames:
    raise FileNotFoundError(f"No score files found for {algorithm}/{processor}")

  combined = pd.concat(frames, ignore_index=True)
  return (combined.groupby("step").agg(
      kl_mean=("kl", "mean"),
      kl_std=("kl", "std"),
      soft_accuracy_mean=("soft_accuracy", "mean"),
      soft_accuracy_std=("soft_accuracy", "std"),
  ).reset_index())


def final_distribution_metrics(root: Path) -> pd.DataFrame:
  """Return final KL and soft-accuracy metrics for all algorithms and sizes."""
  rows = []
  for algorithm in ("bfs_multi", "mst_prim_multi"):
    for processor in PROCESSORS:
      values = []
      for path in score_files(root, algorithm, processor):
        frame = load_score_frame(path)
        last = frame.iloc[-1]
        values.append((float(last["kl"]), float(last["soft_accuracy"])))
      if not values:
        continue
      array = np.asarray(values)
      rows.append({
          "algorithm": algorithm,
          "processor": processor,
          "kl_mean": array[:, 0].mean(),
          "kl_std": array[:, 0].std(ddof=1) if len(array) > 1 else 0.0,
          "soft_accuracy_mean": array[:, 1].mean(),
          "soft_accuracy_std": (array[:, 1].std(ddof=1) if len(array) > 1 else 0.0),
          "num_seeds": len(array),
      })
  return pd.DataFrame(rows)


def sampling_summary(
    root: Path,
    algorithm: str,
    methods: Iterable[str],
    processor: str = "triplet-gmpnn",
) -> pd.DataFrame:
  """Aggregate graph accuracy from n100 sampling reports.

    Each seed contributes the mean of its per-graph ``*_Accuracy`` column.
    The final mean/std are then computed across seeds.
    """
  rows = []
  for size in SIZES:
    per_seed = []
    for path in sampling_files(root, algorithm, size, processor=processor):
      frame = pd.read_csv(path)
      seed_row = {}
      for method in methods:
        for source in SOURCES:
          prefix = f"{method}_{source}"
          column = f"{prefix}_Accuracy"
          if column not in frame.columns:
            raise KeyError(f"Missing {column} in {path}")
          seed_row[f"{prefix}_accuracy"] = frame[column].mean()
      per_seed.append(seed_row)
    if not per_seed:
      continue

    per_seed_frame = pd.DataFrame(per_seed)
    for method in methods:
      for source in SOURCES:
        series = per_seed_frame[f"{method}_{source}_accuracy"]
        rows.append({
            "algorithm": algorithm,
            "processor": processor,
            "size": size,
            "method": method,
            "source": source,
            "metric": "accuracy",
            "mean": series.mean(),
            "std": series.std(ddof=1) if len(series) > 1 else 0.0,
            "num_seeds": len(series),
        })
  return pd.DataFrame(rows)


def _latex_cell(mean: float, std: float, bold: bool = False) -> str:
  mean_str = f"\\bm{{{mean*100:.2f}}}" if bold else f"{mean*100:.2f}"
  std_str = f"{std*100:.2f}"

  return f"${mean_str} \\pm {std_str}$"


def write_sampling_table(
    summary: pd.DataFrame,
    methods: Iterable[str],
    metric: str,
    caption: str,
    label: str,
    output_path: Path,
) -> None:
  """Write a paper-style LaTeX table for graph accuracy."""
  method_list = list(methods)
  sizes = sorted(summary["size"].unique())
  num_methods = len(method_list)
  # Header logic:
  # 1. Multirow in the top row spans down.
  # 2. Bottom row leaves those columns empty.
  lines = [
      "\\begin{table}[hbt!]",
      "    \\centering",
      "    \\small",
      "    \\begin{tabular}{ll" + "c" * num_methods + "}",
      "        \\toprule",
      # Use a literal '*' inside the braces for multirow
      f"        \\multirow{{2}}{{*}}[-2pt]{{\\textbf{{Graph Size}}}} & \\multirow{{2}}{{*}}[-2pt]{{\\textbf{{Distribution}}}} & "
      f"\\multicolumn{{{num_methods}}}{{c}}{{\\textbf{{Extraction Method}}}} \\\\",
      f"        \\cmidrule(lr){{3-{2 + num_methods}}}",
      "        & & " +
      " & ".join(f"\\textbf{{{DISPLAY_NAMES.get(method, method)}}}" for method in method_list) +
      " \\\\",
      "        \\midrule",
  ]
  for i, size in enumerate(sizes):
    # We loop through sources to build the group for each size
    for j, source in enumerate(SOURCES):
      means = []
      for method in method_list:
        row = summary[(summary["size"] == size) & (summary["source"] == source) &
                      (summary["method"] == method) & (summary["metric"] == metric)].iloc[0]
        means.append(float(row["mean"]))

      max_mean = max(means)
      cells = []
      for method, mean in zip(method_list, means):
        row = summary[(summary["size"] == size) & (summary["source"] == source) &
                      (summary["method"] == method) & (summary["metric"] == metric)].iloc[0]
        std = float(row["std"])
        is_best = abs(mean - max_mean) < 1e-12
        cells.append(_latex_cell(mean, std, bold=is_best))

      # LOGIC FOR GROUPING GRAPH SIZE:
      # If it's the first source in the size group, write the multirow.
      # Otherwise, leave the first column empty.
      if j == 0:
        # Assuming SOURCES always has 2 elements (True/Model).
        # If it varies, use len(SOURCES) instead of 2.
        size_col = f"\\multirow{{{len(SOURCES)}}}{{*}}{{$n={size}$}}"
      else:
        size_col = ""

      lines.append(f"        {size_col} & {source} & " + " & ".join(cells) + " \\\\")

    # Add a visual gap between n=5, n=16, etc.
    if i < len(sizes) - 1:
      lines.append("        \\addlinespace")
  lines.extend([
      "        \\bottomrule",
      "    \\end{tabular}",
      f"    \\caption{{{caption}}}",
      f"    \\label{{{label}}}",
      "\\end{table}",
      "",
  ])
  output_path.write_text("\n".join(lines))


def plot_validation_curves(root: Path, output_dir: Path, processor: str = "triplet-gmpnn") -> Path:
  """Plot validation-score learning curves for BFS and Prim."""
  plt = _configure_matplotlib(output_dir)
  fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.0), constrained_layout=True)
  colors = {"bfs_multi": "#1f77b4", "mst_prim_multi": "#d62728"}
  labels = {"bfs_multi": "BFS", "mst_prim_multi": "MST-Prim"}

  for algorithm in ("bfs_multi", "mst_prim_multi"):
    curve = aggregate_learning_curve(root, algorithm, processor)
    x = curve["step"].to_numpy()
    acc_mean = curve["soft_accuracy_mean"].to_numpy()
    acc_std = curve["soft_accuracy_std"].fillna(0).to_numpy()
    color = colors[algorithm]

    ax.plot(x, acc_mean, label=labels[algorithm], color=color, lw=1)
    ax.fill_between(
        x,
        np.maximum(acc_mean - acc_std, 0),
        np.minimum(acc_mean + acc_std, 1),
        color=color,
        alpha=0.16,
        linewidth=0,
    )

  ax.set_xlabel("Training step")
  ax.set_ylabel("Validation score")
  ax.set_ylim(0.88, 1.002)
  ax.grid(True, alpha=0.25)
  ax.legend(loc="lower right")

  output_path = output_dir / "validation-score-curves.pdf"
  fig.savefig(output_path, bbox_inches="tight")
  plt.close(fig)

  return output_path


def plot_train_kl_curves(root: Path, output_dir: Path) -> Path:
  """Plot training KL-divergence curves for BFS and Prim across processors."""
  plt = _configure_matplotlib(output_dir)
  fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.2), constrained_layout=True)
  colors = {
      ("bfs_multi", "triplet-gmpnn"): "#1f77b4",
      ("mst_prim_multi", "triplet-gmpnn"): "#d62728",
      # ("bfs_multi", "pgn"): "#17becf",
      # ("mst_prim_multi", "pgn"): "#ff7f0e",
  }
  labels = {
      ("bfs_multi", "triplet-gmpnn"): "BFS",
      ("mst_prim_multi", "triplet-gmpnn"): "MST-Prim",
      # ("bfs_multi", "pgn"): "BFS, PGN",
      # ("mst_prim_multi", "pgn"): "Prim, PGN",
  }
  rows = []

  for algorithm in ("bfs_multi", "mst_prim_multi"):
    # for processor in PROCESSORS:
    for processor in ("triplet-gmpnn",):  # temporarily only plot triplet-gmpnn curves
      try:
        curve = aggregate_learning_curve(root, algorithm, processor)
      except FileNotFoundError:
        print(f"Skipped KL curve for {algorithm}/{processor}: no score files found")
        continue
      x = curve["step"].to_numpy()
      kl_mean = curve["kl_mean"].to_numpy()
      kl_std = curve["kl_std"].fillna(0).to_numpy()
      color = colors[(algorithm, processor)]

      ax.plot(x, kl_mean, label=labels[(algorithm, processor)], color=color, lw=1)
      ax.fill_between(
          x,
          np.maximum(kl_mean - kl_std, 1e-8),
          np.maximum(kl_mean + kl_std, 1e-8),
          color=color,
          alpha=0.14,
          linewidth=0,
      )
      curve = curve.copy()
      curve["algorithm"] = algorithm
      curve["processor"] = processor
      rows.append(curve)

  ax.set_xlabel("Training step")
  ax.set_ylabel("KL divergence")
  ax.grid(True, alpha=0.25, which="both")
  ax.legend(loc="upper right")

  output_path = output_dir / "kl-curves.pdf"
  fig.savefig(output_path, bbox_inches="tight")
  plt.close(fig)

  if rows:
    pd.concat(rows, ignore_index=True).to_csv(output_dir / "kl-curves.csv", index=False)
  return output_path


def generate_assets(root: Path, output_dir: Path) -> None:
  """Generate all reusable evaluation assets."""
  generate_graph_accuracy_assets(root, output_dir)
  kl_plot_path = plot_train_kl_curves(root, output_dir)
  print(f"Wrote train KL curve to {kl_plot_path}")
  validation_plot_path = plot_validation_curves(root, output_dir)
  print(f"Wrote validation score curve to {validation_plot_path}")


def generate_graph_accuracy_assets(
    root: Path,
    output_dir: Path,
) -> None:
  """Generate n100 graph-accuracy tables from ``*_Accuracy`` columns."""
  output_dir.mkdir(parents=True, exist_ok=True)

  all_summaries = []
  for algorithm, methods in ALGORITHM_METHODS.items():
    for processor in PROCESSORS:
      summary = sampling_summary(root, algorithm, methods, processor=processor)
      if summary.empty:
        print(f"Skipped {algorithm}: no {processor} n100 samples found")
        continue
      all_summaries.append(summary)
      stem = ALGORITHM_OUTPUT_NAMES[algorithm]
      summary.to_csv(output_dir / f"{stem}-{processor}-graph-accuracy-summary.csv", index=False)
      write_sampling_table(
          summary,
          methods,
          "accuracy",
          (f"{ALGORITHM_DISPLAY_NAMES[algorithm]} graph accuracy, "
           "averaged over five seeds."),
          f"tab:{stem}-{processor}-graph-accuracy",
          output_dir / f"{stem}-{processor}-graph-accuracy.tex",
      )

  if all_summaries:
    pd.concat(all_summaries, ignore_index=True).to_csv(output_dir / f"graph-accuracy-summary.csv",
                                                       index=False)
  print(f"Wrote graph accuracy assets to {output_dir}")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--root",
      type=Path,
      default=Path(__file__).resolve().parent / "final",
      help="Directory containing final result runs.",
  )
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=None,
      help="Output directory for generated tables and figures.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  root = args.root.resolve()
  output_dir = args.output_dir
  if output_dir is None:
    output_dir = root / "evaluation_assets"
  generate_assets(root, output_dir.resolve())


if __name__ == "__main__":
  main()
