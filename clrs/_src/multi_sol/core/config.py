"""Configuration dataclasses for multi-solution evaluation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BeamConfig:
  beam_width: int = 3


@dataclass(frozen=True)
class EvaluationConfig:
  beam_width: int = 3

