"""Objective helpers for multi-solution losses."""

import jax
import jax.numpy as jnp

from clrs._src.multi_sol.training.output_head import multisol_softmax


def kl_divergence_truth_pred(truth, pred_logits, epsilon: float = 1e-8):
  """KL divergence utility matching existing MULT_SOL training behavior."""
  return jnp.mean(
      kl_divergence_truth_pred_elementwise(
          truth=truth,
          pred_logits=pred_logits,
          epsilon=epsilon))


def kl_divergence_truth_pred_elementwise(
    truth,
    pred_logits,
    epsilon: float = 1e-8,
):
  """Elementwise KL(truth || pred_probs), shape-compatible with `truth`."""
  probs = multisol_softmax(pred_logits, axis=-1)
  return jax.scipy.special.kl_div(truth, probs + epsilon)
