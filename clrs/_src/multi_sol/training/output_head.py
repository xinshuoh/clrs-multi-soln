"""Output post-processing helpers for multi-solution predictions."""

import jax


def multisol_softmax(logits, axis: int = -1):
  """Convert MULTI_SOLUTION logits to parent-distribution probabilities."""
  return jax.nn.softmax(logits, axis=axis)
