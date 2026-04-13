"""Registries and extension hooks for multi-solution algorithms."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional, Tuple


SpecTransformer = Callable[[Dict[str, Any]], Dict[str, Any]]
SpecFactory = Callable[[Dict[str, Dict[str, Any]]], Dict[str, Any]]
SamplerFactory = Callable[[], type]
Algorithm = Callable[..., Any]
AlgorithmFactory = Callable[[], Algorithm]
EvaluatorFn = Callable[..., dict]


@dataclasses.dataclass(frozen=True)
class MultiSolAlgorithmExtension:
  """Descriptor for one multi-solution algorithm extension."""

  algorithm_name: str
  base_algorithm_name: str
  output_name: str
  spec_transformer: Optional[SpecTransformer] = None
  spec_factory: Optional[SpecFactory] = None
  sampler_factory: Optional[SamplerFactory] = None
  algorithm_factory: Optional[AlgorithmFactory] = None
  evaluator: Optional[EvaluatorFn] = None


_EXTENSIONS: Dict[str, MultiSolAlgorithmExtension] = {}


def ensure_builtin_extensions_registered() -> None:
  # Lazy import to avoid circular dependency at module import time.
  from clrs._src.multi_sol.core import extensions as _extensions  # noqa: F401


def register_extension(extension: MultiSolAlgorithmExtension) -> None:
  name = extension.algorithm_name
  if name in _EXTENSIONS:
    raise ValueError(f"Extension already registered for {name}.")
  _EXTENSIONS[name] = extension


def get_extension(algorithm_name: str) -> Optional[MultiSolAlgorithmExtension]:
  ensure_builtin_extensions_registered()
  return _EXTENSIONS.get(algorithm_name)


def list_extensions() -> Tuple[str, ...]:
  ensure_builtin_extensions_registered()
  return tuple(_EXTENSIONS.keys())


def build_overlay_specs(base_specs_map) -> Dict[str, Dict[str, Any]]:
  """Return extension-generated specs keyed by algorithm name."""
  ensure_builtin_extensions_registered()
  overlays: Dict[str, Dict[str, Any]] = {}
  for extension in _EXTENSIONS.values():
    if extension.spec_factory is not None:
      overlays[extension.algorithm_name] = extension.spec_factory(base_specs_map)
      continue
    if extension.spec_transformer is None:
      continue
    base_spec = base_specs_map[extension.base_algorithm_name]
    overlays[extension.algorithm_name] = extension.spec_transformer(base_spec)
  return overlays


def resolve_specs(base_specs_map) -> Dict[str, Dict[str, Any]]:
  """Return merged map of base specs and extension overlays."""
  ensure_builtin_extensions_registered()
  resolved = dict(base_specs_map)
  resolved.update(build_overlay_specs(base_specs_map))
  return resolved


def get_extension_algorithms(base_algorithm_order) -> Tuple[str, ...]:
  """Append extensions to algorithm order while preserving base ordering."""
  ensure_builtin_extensions_registered()
  ordered = list(base_algorithm_order)
  for name in _EXTENSIONS:
    if name not in ordered:
      ordered.append(name)
  return tuple(ordered)
