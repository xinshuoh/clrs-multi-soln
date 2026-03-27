"""Report building helpers for result logging."""


def build_bfs_result_dict(
    adjacency_flat,
    source_nodes,
    categorical,
    random_sampling,
    prim,
    beam,
):
  return {
      "As": adjacency_flat,
      "Source_Nodes": source_nodes.tolist(),
      "Categorical_Model_Trees": categorical["model_trees"],
      "Categorical_True_Trees": categorical["true_trees"],
      "Categorical_Model_Mask": categorical["model_mask"],
      "Categorical_True_Mask": categorical["true_mask"],
      "Categorical_Model_Accuracy": categorical["model_accuracy"],
      "Categorical_True_Accuracy": categorical["true_accuracy"],
      "Random_Model_Trees": random_sampling["model_trees"],
      "Random_True_Trees": random_sampling["true_trees"],
      "Random_Model_Mask": random_sampling["model_mask"],
      "Random_True_Mask": random_sampling["true_mask"],
      "Random_Model_Accuracy": random_sampling["model_accuracy"],
      "Random_True_Accuracy": random_sampling["true_accuracy"],
      "Prim_Model_Trees": prim["model_trees"],
      "Prim_True_Trees": prim["true_trees"],
      "Prim_Model_Mask": prim["model_mask"],
      "Prim_True_Mask": prim["true_mask"],
      "Prim_Model_Accuracy": prim["model_accuracy"],
      "Prim_True_Accuracy": prim["true_accuracy"],
      "Beam_Model_Trees": beam["model_trees"],
      "Beam_True_Trees": beam["true_trees"],
      "Beam_Model_Mask": beam["model_mask"],
      "Beam_True_Mask": beam["true_mask"],
      "Beam_Model_Accuracy": beam["model_accuracy"],
      "Beam_True_Accuracy": beam["true_accuracy"],
  }

