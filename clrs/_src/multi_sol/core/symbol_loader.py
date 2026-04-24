"""Symbol loading utilities for multi-solution manifest wiring."""

from __future__ import annotations

import importlib


def load_symbol(import_path: str):
  """Load a symbol from `module.submodule:attr_name` import path."""
  if ":" not in import_path:
    raise ValueError(
        f"Invalid symbol path '{import_path}'. Expected 'module.path:attribute'."
    )
  module_name, symbol_name = import_path.split(":", 1)
  if not module_name or not symbol_name:
    raise ValueError(
        f"Invalid symbol path '{import_path}'. Expected 'module.path:attribute'."
    )
  module = importlib.import_module(module_name)
  try:
    return getattr(module, symbol_name)
  except AttributeError as exc:
    raise ValueError(
        f"Symbol '{symbol_name}' not found in module '{module_name}'."
    ) from exc

