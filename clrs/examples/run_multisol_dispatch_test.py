"""Unit tests for optional multi-solution test-time dispatch in run.py."""

import ast
import pathlib

from absl.testing import absltest

from clrs._src.multi_sol.evaluation import artifacts
from clrs._src.multi_sol.evaluation import pipeline


class RunMultisolDispatchTest(absltest.TestCase):

  def _run_flag_defaults(self):
    run_path = pathlib.Path(__file__).resolve().with_name("run.py")
    module = ast.parse(run_path.read_text(encoding="utf-8"))
    defaults = {}
    for node in ast.walk(module):
      if not isinstance(node, ast.Call):
        continue
      if not isinstance(node.func, ast.Attribute):
        continue
      if not isinstance(node.func.value, ast.Name):
        continue
      if node.func.value.id != "flags":
        continue
      if len(node.args) < 2:
        continue
      name_node, default_node = node.args[0], node.args[1]
      if not isinstance(name_node, ast.Constant):
        continue
      if not isinstance(name_node.value, str):
        continue
      if isinstance(default_node, ast.Constant):
        defaults[name_node.value] = default_node.value
    return defaults

  def test_default_profile_uses_fallback(self):
    called = {"fallback": False, "sampling": False}
    original_evaluate_algorithm = pipeline.evaluate_algorithm
    pipeline.evaluate_algorithm = lambda **kwargs: {"score": 1.0}

    def fallback_eval_fn(**kwargs):
      del kwargs
      called["fallback"] = True
      return {"score": 0.5}

    try:
      out = pipeline.evaluate_with_optional_sampling(
          algorithm_name="dfs_multi",
          profile="default",
          multi_sol_algorithm=object(),
          sampler=object(),
          predict_fn=object(),
          sample_count=3,
          rng_key=0,
          extras={},
          artifact_prefix="unused",
          save_artifacts=False,
          fallback_eval_fn=fallback_eval_fn,
      )
    finally:
      pipeline.evaluate_algorithm = original_evaluate_algorithm
    self.assertTrue(called["fallback"])
    self.assertEqual(out["score"], 0.5)

  def test_sampling_profile_uses_sampling(self):
    called = {"fallback": False, "sampling": False}
    seen = {}
    original_evaluate_algorithm = pipeline.evaluate_algorithm
    def _evaluate_algorithm(**kwargs):
      called["sampling"] = True
      seen["save_results_fn"] = kwargs["save_results_fn"]
      return {"score": 0.9}
    pipeline.evaluate_algorithm = _evaluate_algorithm

    def fallback_eval_fn(**kwargs):
      del kwargs
      called["fallback"] = True
      return {"score": 0.1}

    try:
      out = pipeline.evaluate_with_optional_sampling(
          algorithm_name="dfs_multi",
          profile="sampling",
          multi_sol_algorithm=object(),
          sampler=object(),
          predict_fn=object(),
          sample_count=3,
          rng_key=0,
          extras={},
          artifact_prefix="unused",
          save_artifacts=False,
          fallback_eval_fn=fallback_eval_fn,
      )
    finally:
      pipeline.evaluate_algorithm = original_evaluate_algorithm
    self.assertFalse(called["fallback"])
    self.assertTrue(called["sampling"])
    self.assertIs(seen["save_results_fn"], artifacts.discard_report)
    self.assertEqual(out["score"], 0.9)

  def test_sampling_profile_can_enable_artifact_sink(self):
    called = {"fallback": False, "sampling": False}
    seen = {}
    original_evaluate_algorithm = pipeline.evaluate_algorithm
    def _evaluate_algorithm(**kwargs):
      called["sampling"] = True
      seen["save_results_fn"] = kwargs["save_results_fn"]
      return {"score": 0.9}
    pipeline.evaluate_algorithm = _evaluate_algorithm

    def fallback_eval_fn(**kwargs):
      del kwargs
      called["fallback"] = True
      return {"score": 0.1}

    try:
      out = pipeline.evaluate_with_optional_sampling(
          algorithm_name="dfs_multi",
          profile="sampling",
          multi_sol_algorithm=object(),
          sampler=object(),
          predict_fn=object(),
          sample_count=3,
          rng_key=0,
          extras={},
          artifact_prefix="unused",
          save_artifacts=True,
          fallback_eval_fn=fallback_eval_fn,
      )
    finally:
      pipeline.evaluate_algorithm = original_evaluate_algorithm
    self.assertFalse(called["fallback"])
    self.assertTrue(called["sampling"])
    self.assertIs(seen["save_results_fn"], artifacts.save_pickle_report)
    self.assertEqual(out["score"], 0.9)

  def test_sampling_profile_uses_custom_report_sink(self):
    called = {"fallback": False, "sampling": False}

    def custom_sink(_result_dict, _filename):
      del _result_dict, _filename

    seen = {}
    original_evaluate_algorithm = pipeline.evaluate_algorithm
    def _evaluate_algorithm(**kwargs):
      called["sampling"] = True
      seen["save_results_fn"] = kwargs["save_results_fn"]
      return {"score": 0.8}
    pipeline.evaluate_algorithm = _evaluate_algorithm

    def fallback_eval_fn(**kwargs):
      del kwargs
      called["fallback"] = True
      return {"score": 0.1}

    try:
      out = pipeline.evaluate_with_optional_sampling(
          algorithm_name="dfs_multi",
          profile="sampling",
          multi_sol_algorithm=object(),
          sampler=object(),
          predict_fn=object(),
          sample_count=3,
          rng_key=0,
          extras={},
          artifact_prefix="unused",
          save_artifacts=False,
          fallback_eval_fn=fallback_eval_fn,
          report_sink=custom_sink,
      )
    finally:
      pipeline.evaluate_algorithm = original_evaluate_algorithm
    self.assertFalse(called["fallback"])
    self.assertTrue(called["sampling"])
    self.assertIs(seen["save_results_fn"], custom_sink)
    self.assertEqual(out["score"], 0.8)

  def test_sampling_profile_forwards_sampling_kwargs(self):
    called = {"fallback": False, "sampling": False}
    seen = {}
    original_evaluate_algorithm = pipeline.evaluate_algorithm
    def _evaluate_algorithm(**kwargs):
      seen["filename"] = kwargs["filename"]
      seen["NSE"] = kwargs["NSE"]
      called["sampling"] = True
      return {"score": 0.7}
    pipeline.evaluate_algorithm = _evaluate_algorithm

    def fallback_eval_fn(**kwargs):
      del kwargs
      called["fallback"] = True
      return {"score": 0.1}

    try:
      out = pipeline.evaluate_with_optional_sampling(
          algorithm_name="dfs_multi",
          profile="sampling",
          multi_sol_algorithm=object(),
          sampler=object(),
          predict_fn=object(),
          sample_count=3,
          rng_key=0,
          extras={},
          artifact_prefix="unused",
          save_artifacts=False,
          fallback_eval_fn=fallback_eval_fn,
          sampling_kwargs={"NSE": 17, "vd_flag": True},
      )
    finally:
      pipeline.evaluate_algorithm = original_evaluate_algorithm
    self.assertFalse(called["fallback"])
    self.assertTrue(called["sampling"])
    self.assertEqual(out["score"], 0.7)
    self.assertEqual(seen["NSE"], 17)
    self.assertEqual(seen["filename"], "unused_dfs_multi")

  def test_sampling_profile_falls_back_for_non_multisol_algorithm(self):
    called = {"fallback": False}
    seen = {}
    fallback_kwargs = {}

    sampler = object()
    predict_fn = object()
    extras = {"examples_seen": 7}

    def fallback_eval_fn(**kwargs):
      called["fallback"] = True
      fallback_kwargs.update(kwargs)
      return {"score": 0.5}

    out = pipeline.evaluate_with_optional_sampling(
        algorithm_name="bfs",
        profile="sampling",
        multi_sol_algorithm=None,
        sampler=sampler,
        predict_fn=predict_fn,
        sample_count=3,
        rng_key=0,
        extras=extras,
        artifact_prefix="sampling_eval",
        save_artifacts=False,
        fallback_eval_fn=fallback_eval_fn,
    )

    self.assertTrue(called["fallback"])
    self.assertEqual(
        set(fallback_kwargs.keys()),
        {"sampler", "predict_fn", "sample_count", "rng_key", "extras"},
    )
    self.assertIs(fallback_kwargs["sampler"], sampler)
    self.assertIs(fallback_kwargs["predict_fn"], predict_fn)
    self.assertEqual(fallback_kwargs["sample_count"], 3)
    self.assertEqual(fallback_kwargs["rng_key"], 0)
    self.assertIs(fallback_kwargs["extras"], extras)
    self.assertEqual(out["score"], 0.5)

  def test_default_profile_skips_sampling_algorithm(self):
    called = {"fallback": False}
    original_evaluate_algorithm = pipeline.evaluate_algorithm
    def _evaluate_algorithm(**kwargs):
      raise AssertionError("Sampling path should not run for default profile.")
    pipeline.evaluate_algorithm = _evaluate_algorithm

    def fallback_eval_fn(**kwargs):
      del kwargs
      called["fallback"] = True
      return {"score": 0.4}

    try:
      out = pipeline.evaluate_with_optional_sampling(
          algorithm_name="dfs_multi",
          profile="default",
          multi_sol_algorithm=object(),
          sampler=object(),
          predict_fn=object(),
          sample_count=3,
          rng_key=0,
          extras={},
          artifact_prefix="sampling_eval",
          save_artifacts=False,
          fallback_eval_fn=fallback_eval_fn,
      )
    finally:
      pipeline.evaluate_algorithm = original_evaluate_algorithm

    self.assertTrue(called["fallback"])
    self.assertEqual(out["score"], 0.4)

  def test_run_flag_defaults_preserve_upstream_behavior(self):
    defaults = self._run_flag_defaults()
    self.assertEqual(defaults["evaluation_profile"], "default")
    self.assertEqual(defaults["val_evaluation_profile"], "default")
    self.assertEqual(defaults["run_mode"], "train_eval")
    self.assertIs(defaults["save_sampling_artifacts"], False)
    self.assertEqual(defaults["sampling_artifact_prefix"], "sampling_eval")
    self.assertEqual(defaults["run_dir"], "")
    self.assertEqual(defaults["filename"], "")
    self.assertIs(defaults["results_df"], False)
    self.assertIs(defaults["save_df"], False)
    self.assertIs(defaults["save_model_to_file"], False)
    self.assertEqual(defaults["model_filename"], "")
    self.assertEqual(defaults["load_models_from_dir"], "")
    self.assertIs(defaults["validate_distributions"], False)
    self.assertEqual(defaults["NSE"], 25)
    self.assertEqual(defaults["distribution_validation_graphs"], 0)
    self.assertIsNone(defaults["test_length"])
    self.assertEqual(defaults["test_dataset_cache_dir"], "")
    self.assertEqual(defaults["test_dataset_seed"], 0)
    self.assertEqual(defaults["test_num_samples"], 0)
    self.assertIs(defaults["refresh_test_dataset_cache"], False)

  def test_run_source_resolves_checkpoint_default_under_run_dir(self):
    run_path = pathlib.Path(__file__).resolve().with_name("run.py")
    source = run_path.read_text(encoding="utf-8")
    self.assertIn("def _resolve_checkpoint_path", source)
    self.assertIn("checkpoint_flag = FLAGS['checkpoint_path']", source)
    self.assertIn("if checkpoint_flag.present", source)
    self.assertIn("os.path.join(run_dir, 'checkpoints')", source)

  def test_run_source_supports_train_eval_model_layout(self):
    run_path = pathlib.Path(__file__).resolve().with_name("run.py")
    source = run_path.read_text(encoding="utf-8")
    self.assertIn("def _resolve_model_load_path", source)
    self.assertIn("FLAGS.load_models_from_dir", source)
    self.assertIn("f'seed_{seed}'", source)
    self.assertIn("FLAGS.run_mode == 'train'", source)
    self.assertIn("create_train_samplers=FLAGS.run_mode != 'eval'", source)
    self.assertIn("FLAGS.chunked_training and FLAGS.run_mode != 'eval'", source)
    self.assertIn("return []", source)

  def test_run_source_supports_cached_positive_length_test_datasets(self):
    run_path = pathlib.Path(__file__).resolve().with_name("run.py")
    source = run_path.read_text(encoding="utf-8")
    self.assertIn("def _cached_test_dataset_path", source)
    self.assertIn("def _load_or_create_cached_test_feedback", source)
    self.assertIn("FLAGS.test_dataset_cache_dir", source)
    self.assertIn("FLAGS.test_dataset_seed", source)
    self.assertIn("FLAGS.test_num_samples", source)
    self.assertIn("FLAGS.refresh_test_dataset_cache", source)
    self.assertIn("_iterate_cached_feedback(feedback, batch_size)", source)


if __name__ == "__main__":
  absltest.main()
