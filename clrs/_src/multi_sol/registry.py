"""Metadata registry for multi-solution algorithms."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from clrs._src.multi_sol.core.definitions import MultiSolAlgorithm


_EXTENSIONS: Dict[str, MultiSolAlgorithm] = {}
_BUILTINS_REGISTERED = False


def _builtin_definitions() -> Tuple[MultiSolAlgorithm, ...]:
  # Lazy imports avoid circular import with clrs._src.samplers.
  from clrs._src.multi_sol.algorithms.bellman_ford import definition as bellman_ford
  from clrs._src.multi_sol.algorithms.bfs import definition as bfs
  from clrs._src.multi_sol.algorithms.dfs import definition as dfs
  from clrs._src.multi_sol.algorithms.mst_prim import definition as mst_prim
  return (
      dfs.DEFINITION,
      bfs.DEFINITION,
      bellman_ford.DEFINITION,
      mst_prim.DEFINITION,
  )


def ensure_builtin_extensions_registered() -> None:
  global _BUILTINS_REGISTERED
  if _BUILTINS_REGISTERED:
    return
  for definition in _builtin_definitions():
    register_extension(definition)
  _BUILTINS_REGISTERED = True


def register_extension(extension: MultiSolAlgorithm) -> None:
  name = extension.name
  if name in _EXTENSIONS:
    raise ValueError(f"Extension already registered for {name}.")
  _EXTENSIONS[name] = extension


def get(algorithm_name: str) -> Optional[MultiSolAlgorithm]:
  ensure_builtin_extensions_registered()
  return _EXTENSIONS.get(algorithm_name)


def names() -> Tuple[str, ...]:
  ensure_builtin_extensions_registered()
  return tuple(_EXTENSIONS.keys())


__all__ = (
    "get",
    "names",
    "register_extension",
)
