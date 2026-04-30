"""Load multi-solution extension descriptors from TOML manifests."""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Iterable, Tuple

from clrs._src.multi_sol.core import definitions
from clrs._src.multi_sol.core import manifest_schema
from clrs._src.multi_sol.core import symbol_loader
from clrs._src.multi_sol import specs as multi_sol_specs


def _default_manifest_dir() -> Path:
  # multi_sol/core/manifest_loader.py -> multi_sol/manifests/
  return Path(__file__).resolve().parents[1] / "manifests"


def _iter_manifest_paths(manifest_dir: Path | None = None) -> Iterable[Path]:
  directory = manifest_dir or _default_manifest_dir()
  if not directory.exists():
    return ()
  return sorted(directory.glob("*.toml"))


def _load_manifest(path: Path) -> manifest_schema.MultiSolManifest:
  with path.open("rb") as fh:
    raw = tomllib.load(fh)
  try:
    return manifest_schema.MultiSolManifest.from_dict(raw)
  except ValueError as exc:
    raise ValueError(f"Invalid manifest {path}: {exc}") from exc


def _build_extension(
    manifest: manifest_schema.MultiSolManifest,
) -> definitions.MultiSolAlgorithm:
  try:
    spec_provider = dict(multi_sol_specs.MULTI_SOL_SPECS[manifest.algorithm_name])
  except KeyError as exc:
    raise ValueError(
        f"Spec not found for algorithm '{manifest.algorithm_name}' in "
        "clrs._src.multi_sol.specs.MULTI_SOL_SPECS."
    ) from exc
  sampler_cls = symbol_loader.load_symbol(manifest.sampler_class_path)
  algorithm_callable = symbol_loader.load_symbol(manifest.algorithm_callable_path)
  evaluator_callable = (
      symbol_loader.load_symbol(manifest.evaluator_callable_path)
      if manifest.evaluator_callable_path is not None
      else None
  )

  return definitions.MultiSolAlgorithm(
      algorithm_name=manifest.algorithm_name,
      base_algorithm_name=manifest.base_algorithm_name,
      spec=spec_provider,
      sampler_class=sampler_cls,
      algorithm=algorithm_callable,
      evaluator=evaluator_callable,
  )


def load_manifest_extensions(
    manifest_dir: Path | None = None,
) -> Tuple[definitions.MultiSolAlgorithm, ...]:
  """Load all extensions declared in TOML manifests."""
  return tuple(
      _build_extension(_load_manifest(path))
      for path in _iter_manifest_paths(manifest_dir)
  )
