"""Output-type policy registry for multi-solution training semantics.

Prefer `specs.Type.MULTI_SOLUTION` for new type policies. `specs.Type.MULT_SOL`
is still accepted as a compatibility alias while multi-solution parity tests
are being finalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax.numpy as jnp

from clrs._src import specs
from clrs._src.multi_sol.training import objectives
from clrs._src.multi_sol.training import output_head


_NodeDecoderFactory = Callable[[Callable[..., Any], int, int], Tuple[Any, ...]]
_NodeDecodeFn = Callable[
    [Sequence[Any], Any, Any, Any, bool, bool], Any
]
_PostprocessFn = Callable[[Any, bool], Tuple[Any, str]]
_LossFn = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class TypePolicy:
  """Custom behavior for one output type."""

  construct_node_decoders: Optional[_NodeDecoderFactory] = None
  decode_node_logits: Optional[_NodeDecodeFn] = None
  postprocess: Optional[_PostprocessFn] = None
  output_loss: Optional[_LossFn] = None
  output_loss_elementwise: Optional[_LossFn] = None
  hint_loss_elementwise: Optional[_LossFn] = None


_TYPE_POLICIES: Dict[str, TypePolicy] = {}


def register_type_policy(type_name: str, policy: TypePolicy) -> None:
  if type_name in _TYPE_POLICIES:
    raise ValueError(f"Policy already registered for type {type_name}.")
  _TYPE_POLICIES[type_name] = policy


def get_type_policy(type_name: str) -> Optional[TypePolicy]:
  return _TYPE_POLICIES.get(type_name)


def construct_node_decoders_for_type(
    type_name: str,
    linear: Callable[..., Any],
    hidden_dim: int,
    nb_dims: int,
) -> Optional[Tuple[Any, ...]]:
  policy = get_type_policy(type_name)
  if policy is None or policy.construct_node_decoders is None:
    return None
  return policy.construct_node_decoders(linear, hidden_dim, nb_dims)


def decode_node_logits_for_type(
    type_name: str,
    decoders,
    h_t,
    edge_fts,
    adj_mat,
    inf_bias: bool,
    repred: bool,
):
  policy = get_type_policy(type_name)
  if policy is None or policy.decode_node_logits is None:
    return None
  return policy.decode_node_logits(
      decoders, h_t, edge_fts, adj_mat, inf_bias, repred)


def postprocess_for_type(type_name: str, data, hard: bool):
  policy = get_type_policy(type_name)
  if policy is None or policy.postprocess is None:
    return None
  return policy.postprocess(data, hard)


def output_loss_for_type(type_name: str, truth_data, pred_logits):
  policy = get_type_policy(type_name)
  if policy is None or policy.output_loss is None:
    return None
  return policy.output_loss(truth_data, pred_logits)


def output_loss_elementwise_for_type(type_name: str, truth_data, pred_logits):
  policy = get_type_policy(type_name)
  if policy is None or policy.output_loss_elementwise is None:
    return None
  return policy.output_loss_elementwise(truth_data, pred_logits)


def hint_loss_elementwise_for_type(type_name: str, truth_data, pred_logits):
  policy = get_type_policy(type_name)
  if policy is None or policy.hint_loss_elementwise is None:
    return None
  return policy.hint_loss_elementwise(truth_data, pred_logits)


def registered_types() -> Tuple[str, ...]:
  return tuple(_TYPE_POLICIES.keys())


def _multisol_construct_node_decoders(
    linear: Callable[..., Any],
    hidden_dim: int,
    nb_dims: int,
) -> Tuple[Any, ...]:
  del nb_dims
  return (linear(hidden_dim), linear(hidden_dim), linear(hidden_dim),
          linear(1))


def _multisol_decode_node_logits(
    decoders,
    h_t,
    edge_fts,
    adj_mat,
    inf_bias: bool,
    repred: bool,
):
  del repred
  p_1 = decoders[0](h_t)  # from vector
  p_2 = decoders[1](h_t)  # to vector
  p_3 = decoders[2](edge_fts)  # edge features

  p_e = jnp.expand_dims(p_2, -2) + p_3
  p_m = jnp.maximum(jnp.expand_dims(p_1, -2), jnp.transpose(p_e, (0, 2, 1, 3)))
  preds = jnp.squeeze(decoders[3](p_m), -1)

  if inf_bias:
    per_batch_min = jnp.min(preds, axis=range(1, preds.ndim), keepdims=True)
    preds = jnp.where(adj_mat > 0.5, preds, jnp.minimum(-1.0, per_batch_min - 1.0))
  return preds


def _multisol_postprocess(data, hard: bool):
  del hard
  return output_head.multisol_softmax(data, axis=-1), specs.Type.MULTI_SOLUTION


def _multisol_output_loss(truth_data, pred_logits):
  return objectives.kl_divergence_truth_pred(truth_data, pred_logits)


def _multisol_output_loss_elementwise(truth_data, pred_logits):
  return objectives.kl_divergence_truth_pred_elementwise(truth_data, pred_logits)


register_type_policy(
    specs.Type.MULTI_SOLUTION,
    TypePolicy(
        construct_node_decoders=_multisol_construct_node_decoders,
        decode_node_logits=_multisol_decode_node_logits,
        postprocess=_multisol_postprocess,
        output_loss=_multisol_output_loss,
        output_loss_elementwise=_multisol_output_loss_elementwise,
        hint_loss_elementwise=_multisol_output_loss_elementwise,
    ),
)
