"""Small registries for extraction and validation components."""

from typing import Callable, Dict, TypeVar

T = TypeVar("T")


class Registry:
  """Name -> factory registry."""

  def __init__(self):
    self._factories: Dict[str, Callable[[], T]] = {}

  def register(self, name: str, factory: Callable[[], T]) -> None:
    if name in self._factories:
      raise ValueError(f"Duplicate registry key: {name}")
    self._factories[name] = factory

  def create(self, name: str) -> T:
    if name not in self._factories:
      raise KeyError(f"Unknown registry key: {name}")
    return self._factories[name]()

  def keys(self):
    return tuple(self._factories.keys())

