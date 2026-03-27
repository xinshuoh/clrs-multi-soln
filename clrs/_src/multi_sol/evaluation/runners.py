"""Algorithm-agnostic evaluation runners for batch extraction/validation."""

from typing import Callable, Dict

from clrs._src.multi_sol.evaluation import metrics


def evaluate_sampling_pair(
    model_sample_fn: Callable,
    true_sample_fn: Callable,
    model_input,
    true_input,
    adjacency,
    source_nodes,
    validate_fn: Callable,
    **kwargs,
) -> Dict[str, object]:
  model_trees = model_sample_fn(model_input, **kwargs)
  true_trees = true_sample_fn(true_input, **kwargs)
  model_mask = [
      validate_fn(adjacency[i], model_trees[i], int(source_nodes[i]))
      for i in range(len(model_trees))
  ]
  true_mask = [
      validate_fn(adjacency[i], true_trees[i], int(source_nodes[i]))
      for i in range(len(true_trees))
  ]
  return {
      "model_trees": model_trees,
      "true_trees": true_trees,
      "model_mask": model_mask,
      "true_mask": true_mask,
      "model_accuracy": metrics.accuracy(model_mask),
      "true_accuracy": metrics.accuracy(true_mask),
  }

