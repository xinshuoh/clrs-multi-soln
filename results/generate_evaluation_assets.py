"""Generate dissertation evaluation tables and figures from final result CSVs.

This script is intentionally reusable: it centralises the aggregation logic used
for the dissertation evaluation chapter instead of relying on ad hoc notebooks or
one-off shell snippets.

Usage:
  python results/generate_evaluation_assets.py
  python results/generate_evaluation_assets.py --root results/final_for_diss
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
BFS_METHODS = ("Categorical", "Prim", "Beam", "Random")
PRIM_METHODS = ("Argmax", "Tree", "Greedy", "Random")
SOURCES = ("True", "Model")
DISPLAY_NAMES = {
    "Prim": "Prim-like",
}


def _configure_matplotlib(output_dir: Path):
    """Import and configure matplotlib with a writable cache directory."""
    cache_dir = output_dir / ".matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
        }
    )
    return plt


def score_files(root: Path, algorithm: str, size: int) -> list[Path]:
    """Return score-result files for an algorithm and evaluation size."""
    if algorithm == "bfs_multi":
        pattern = root / f"bfs_multi_{size}_5seeds" / "seed_*" / "score-results.csv"
    elif algorithm == "mst_prim_multi":
        pattern = root / f"mst_prim_multi_{size}_seed*" / "score-results.csv"
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return [Path(p) for p in sorted(glob.glob(str(pattern)))]


def sampling_files(root: Path, algorithm: str, size: int) -> list[Path]:
    """Return per-seed sampling report files for an algorithm and size."""
    if algorithm == "bfs_multi":
        pattern = (
            root / f"bfs_multi_{size}_5seeds" / "seed_*" / f"bfs_multi_{size}*_BFS.csv"
        )
    elif algorithm == "mst_prim_multi":
        pattern = (
            root / f"mst_prim_multi_{size}_seed*" / f"mst_prim_multi_{size}_MSTPrim.csv"
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
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
        }
    )


def aggregate_learning_curve(root: Path, algorithm: str, size: int) -> pd.DataFrame:
    """Aggregate KL and soft-accuracy curves across seeds."""
    frames = []
    for seed_idx, path in enumerate(score_files(root, algorithm, size)):
        frame = load_score_frame(path)
        frame = frame[["step", "kl", "soft_accuracy"]].copy()
        frame["seed_idx"] = seed_idx
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No score files found for {algorithm}, n={size}")

    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby("step")
        .agg(
            kl_mean=("kl", "mean"),
            kl_std=("kl", "std"),
            soft_accuracy_mean=("soft_accuracy", "mean"),
            soft_accuracy_std=("soft_accuracy", "std"),
        )
        .reset_index()
    )


def final_distribution_metrics(root: Path) -> pd.DataFrame:
    """Return final KL and soft-accuracy metrics for all algorithms and sizes."""
    rows = []
    for algorithm in ("bfs_multi", "mst_prim_multi"):
        for size in SIZES:
            values = []
            for path in score_files(root, algorithm, size):
                frame = load_score_frame(path)
                last = frame.iloc[-1]
                values.append((float(last["kl"]), float(last["soft_accuracy"])))
            if not values:
                continue
            array = np.asarray(values)
            rows.append(
                {
                    "algorithm": algorithm,
                    "size": size,
                    "kl_mean": array[:, 0].mean(),
                    "kl_std": array[:, 0].std(ddof=1) if len(array) > 1 else 0.0,
                    "soft_accuracy_mean": array[:, 1].mean(),
                    "soft_accuracy_std": (
                        array[:, 1].std(ddof=1) if len(array) > 1 else 0.0
                    ),
                    "num_seeds": len(array),
                }
            )
    return pd.DataFrame(rows)


def sampling_summary(
    root: Path,
    algorithm: str,
    methods: Iterable[str],
) -> pd.DataFrame:
    """Aggregate graph-accuracy and diversity metrics from sampling reports."""
    rows = []
    for size in SIZES:
        per_seed = []
        for path in sampling_files(root, algorithm, size):
            frame = pd.read_csv(path)
            seed_row = {}
            for method in methods:
                for source in SOURCES:
                    prefix = f"{method}_{source}"
                    seed_row[f"{prefix}_valid"] = frame[f"{prefix}_Valids"].mean()
                    seed_row[f"{prefix}_unique"] = frame[f"{prefix}_Uniques"].mean()
                    seed_row[f"{prefix}_valid_unique"] = frame[
                        f"{prefix}_Valids_Uniques"
                    ].mean()
            per_seed.append(seed_row)
        if not per_seed:
            continue

        per_seed_frame = pd.DataFrame(per_seed)
        for method in methods:
            for source in SOURCES:
                for metric in ("valid", "unique", "valid_unique"):
                    series = per_seed_frame[f"{method}_{source}_{metric}"]
                    rows.append(
                        {
                            "algorithm": algorithm,
                            "size": size,
                            "method": method,
                            "source": source,
                            "metric": metric,
                            "mean": series.mean(),
                            "std": series.std(ddof=1) if len(series) > 1 else 0.0,
                            "num_seeds": len(series),
                        }
                    )
    return pd.DataFrame(rows)


def _latex_cell(mean: float, std: float) -> str:
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def write_sampling_table(
    summary: pd.DataFrame,
    methods: Iterable[str],
    metric: str,
    caption: str,
    label: str,
    output_path: Path,
) -> None:
    """Write a paper-style LaTeX table for graph accuracy or diversity."""
    method_list = list(methods)
    lines = [
        "\\begin{table}[hbt!]",
        "    \\centering",
        "    \\small",
        "    \\begin{tabular}{ll" + "c" * len(method_list) + "}",
        "        \\toprule",
        "        \\textbf{Size} & \\textbf{Distribution} & "
        + " & ".join(
            f"\\textbf{{{DISPLAY_NAMES.get(method, method)}}}" for method in method_list
        )
        + " \\\\",
        "        \\midrule",
    ]
    for size in SIZES:
        for source in SOURCES:
            cells = []
            for method in method_list:
                row = summary[
                    (summary["size"] == size)
                    & (summary["source"] == source)
                    & (summary["method"] == method)
                    & (summary["metric"] == metric)
                ].iloc[0]
                cells.append(_latex_cell(float(row["mean"]), float(row["std"])))
            lines.append(
                f"        $n={size}$ & {source} & " + " & ".join(cells) + " \\\\"
            )
    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            f"    \\caption{{{caption}}}",
            f"    \\label{{{label}}}",
            "\\end{table}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines))


def plot_validation_curves(root: Path, output_dir: Path, size: int = 16) -> Path:
    """Plot validation-score learning curves for BFS and Prim."""
    plt = _configure_matplotlib(output_dir)
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.0), constrained_layout=True)
    colors = {"bfs_multi": "#1f77b4", "mst_prim_multi": "#d62728"}
    labels = {"bfs_multi": "BFS-Multi", "mst_prim_multi": "Prim-Multi"}

    for algorithm in ("bfs_multi", "mst_prim_multi"):
        curve = aggregate_learning_curve(root, algorithm, size)
        x = curve["step"].to_numpy()
        acc_mean = curve["soft_accuracy_mean"].to_numpy()
        acc_std = curve["soft_accuracy_std"].fillna(0).to_numpy()
        color = colors[algorithm]

        ax.plot(x, acc_mean, label=labels[algorithm], color=color, lw=1.8)
        ax.fill_between(
            x,
            np.maximum(acc_mean - acc_std, 0),
            np.minimum(acc_mean + acc_std, 1),
            color=color,
            alpha=0.16,
            linewidth=0,
        )

    ax.set_title("Validation Score")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation score")
    ax.set_ylim(0.88, 1.002)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=False)

    output_path = output_dir / "validation-learning-curves.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_assets(root: Path, output_dir: Path) -> None:
    """Generate all reusable evaluation assets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    validation = final_distribution_metrics(root)
    validation.to_csv(output_dir / "distribution-fitting-validation.csv", index=False)

    bfs = sampling_summary(root, "bfs_multi", BFS_METHODS)
    bfs.to_csv(output_dir / "bfs-sampling-summary.csv", index=False)
    write_sampling_table(
        bfs,
        BFS_METHODS,
        "valid",
        "BFS-Multi graph accuracy on test graphs of size $5$, $16$, and $64$.",
        "tab:bfs-graph-accuracy",
        output_dir / "bfs-graph-accuracy.tex",
    )
    write_sampling_table(
        bfs,
        BFS_METHODS,
        "unique",
        "BFS-Multi diversity, measured as the proportion of distinct predecessor arrays among 25 samples.",
        "tab:bfs-diversity",
        output_dir / "bfs-diversity.tex",
    )

    prim = sampling_summary(root, "mst_prim_multi", PRIM_METHODS)
    prim.to_csv(output_dir / "prim-sampling-summary.csv", index=False)
    write_sampling_table(
        prim,
        PRIM_METHODS,
        "valid",
        "Prim-Multi graph accuracy on test graphs of size $5$, $16$, and $64$.",
        "tab:prim-graph-accuracy",
        output_dir / "prim-graph-accuracy.tex",
    )
    write_sampling_table(
        prim,
        PRIM_METHODS,
        "unique",
        "Prim-Multi diversity, measured as the proportion of distinct predecessor arrays among 25 samples.",
        "tab:prim-diversity",
        output_dir / "prim-diversity.tex",
    )

    plot_path = plot_validation_curves(root, output_dir)
    print(f"Wrote evaluation assets to {output_dir}")
    print(f"Wrote validation curve to {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "final_for_diss",
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
