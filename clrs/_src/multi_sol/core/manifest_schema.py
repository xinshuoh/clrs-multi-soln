"""Schema validation for multi-solution extension manifests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


def _required_str(raw: Dict[str, Any], key: str, *, where: str = "manifest") -> str:
  value = raw.get(key)
  if not isinstance(value, str) or not value:
    raise ValueError(f"Missing or invalid '{key}' in {where}.")
  return value


def _optional_required_str(raw: Dict[str, Any] | None, key: str, *, where: str) -> str | None:
  if raw is None:
    return None
  return _required_str(raw, key, where=where)


@dataclass(frozen=True)
class MultiSolManifest:
  """Validated manifest payload for one multi-solution extension."""

  algorithm_name: str
  base_algorithm_name: str
  sampler_class_path: str
  algorithm_callable_path: str
  evaluator_callable_path: str | None

  @classmethod
  def from_dict(cls, raw: Dict[str, Any]) -> "MultiSolManifest":
    if not isinstance(raw, dict):
      raise ValueError("Manifest content must be a table/dictionary.")

    sampler = raw.get("sampler")
    algorithm = raw.get("algorithm")
    evaluator = raw.get("evaluator")
    if not isinstance(sampler, dict):
      raise ValueError("Missing or invalid [sampler] table in manifest.")
    if not isinstance(algorithm, dict):
      raise ValueError("Missing or invalid [algorithm] table in manifest.")
    if evaluator is not None and not isinstance(evaluator, dict):
      raise ValueError("Missing or invalid [evaluator] table in manifest.")

    return cls(
        algorithm_name=_required_str(raw, "algorithm_name"),
        base_algorithm_name=_required_str(raw, "base_algorithm_name"),
        sampler_class_path=_required_str(
            sampler, "class", where="[sampler]"),
        algorithm_callable_path=_required_str(
            algorithm, "callable", where="[algorithm]"),
        evaluator_callable_path=_optional_required_str(
            evaluator, "callable", where="[evaluator]"),
    )
