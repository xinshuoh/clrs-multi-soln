"""Registries and extension hooks for multi-solution algorithms."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from clrs._src.multi_sol.core.definitions import MultiSolAlgorithmDefinition
from clrs._src.multi_sol.core.definitions import MultiSolAlgorithmExtension
from clrs._src.multi_sol.core.definitions import MultiSolExtensionDefinition


_EXTENSIONS: Dict[str, MultiSolExtensionDefinition] = {}
_BUILTINS_REGISTERED = False


def ensure_builtin_extensions_registered() -> None:
  global _BUILTINS_REGISTERED
  if _BUILTINS_REGISTERED:
    return
  # Lazy import to avoid circular dependency at module import time.
  from clrs._src.multi_sol.algorithms import builtins
  for definition in builtins.BUILTIN_DEFINITIONS:
    register_extension(definition)
  _BUILTINS_REGISTERED = True


def register_extension(extension: MultiSolAlgorithmDefinition) -> None:
  name = extension.algorithm_name
  if name in _EXTENSIONS:
    raise ValueError(f"Extension already registered for {name}.")
  _EXTENSIONS[name] = extension


def get_extension(algorithm_name: str) -> Optional[MultiSolExtensionDefinition]:
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
    if callable(extension.spec):
      overlays[extension.algorithm_name] = extension.spec(base_specs_map)
    else:
      overlays[extension.algorithm_name] = dict(extension.spec)
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
