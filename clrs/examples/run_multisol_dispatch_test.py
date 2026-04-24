"""Unit tests for optional multi-solution test-time dispatch in run.py."""

import ast
import pathlib

from absl.testing import absltest

from clrs._src.multi_sol.evaluation import dispatch
from clrs._src.multi_sol.evaluation import reporting


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
    called = {"fallback": False, "extension": False}

    def extension_evaluator(**kwargs):
      del kwargs
      called["extension"] = True
      return {"score": 1.0}

    def fallback_eval_fn(**kwargs):
      del kwargs
      called["fallback"] = True
      return {"score": 0.5}

    out = dispatch.evaluate_with_optional_extension(
        algorithm_name="dfs_multi",
        profile="default",
        extension_evaluator=extension_evaluator,
        sampler=object(),
        predict_fn=object(),
        sample_count=3,
        rng_key=0,
        extras={},
        artifact_prefix="unused",
        save_artifacts=False,
        fallback_eval_fn=fallback_eval_fn,
    )
    self.assertTrue(called["fallback"])
    self.assertFalse(called["extension"])
    self.assertEqual(out["score"], 0.5)

  def test_sampling_profile_uses_extension(self):
    called = {"fallback": False, "extension": False}

    def extension_evaluator(**kwargs):
      called["extension"] = True
      self.assertIs(kwargs["save_results_fn"], reporting.discard_report)
      return {"score": 0.9}

    def fallback_eval_fn(**kwargs):
      del kwargs
      called["fallback"] = True
      return {"score": 0.1}

    out = dispatch.evaluate_with_optional_extension(
        algorithm_name="dfs_multi",
        profile="sampling",
        extension_evaluator=extension_evaluator,
        sampler=object(),
        predict_fn=object(),
        sample_count=3,
        rng_key=0,
        extras={},
        artifact_prefix="unused",
        save_artifacts=False,
        fallback_eval_fn=fallback_eval_fn,
    )
    self.assertFalse(called["fallback"])
    self.assertTrue(called["extension"])
    self.assertEqual(out["score"], 0.9)

  def test_sampling_profile_can_enable_artifact_sink(self):
    called = {"fallback": False, "extension": False}

    def extension_evaluator(**kwargs):
      called["extension"] = True
      self.assertIs(kwargs["save_results_fn"], reporting.save_pickle_report)
      return {"score": 0.9}

    def fallback_eval_fn(**kwargs):
      del kwargs
      called["fallback"] = True
      return {"score": 0.1}

    out = dispatch.evaluate_with_optional_extension(
        algorithm_name="dfs_multi",
        profile="sampling",
        extension_evaluator=extension_evaluator,
        sampler=object(),
        predict_fn=object(),
        sample_count=3,
        rng_key=0,
        extras={},
        artifact_prefix="unused",
        save_artifacts=True,
        fallback_eval_fn=fallback_eval_fn,
    )
    self.assertFalse(called["fallback"])
    self.assertTrue(called["extension"])
    self.assertEqual(out["score"], 0.9)

  def test_sampling_profile_uses_custom_report_sink(self):
    called = {"fallback": False, "extension": False}

    def custom_sink(_result_dict, _filename):
      del _result_dict, _filename

    def extension_evaluator(**kwargs):
      called["extension"] = True
      self.assertIs(kwargs["save_results_fn"], custom_sink)
      return {"score": 0.8}

    def fallback_eval_fn(**kwargs):
      del kwargs
      called["fallback"] = True
      return {"score": 0.1}

    out = dispatch.evaluate_with_optional_extension(
        algorithm_name="dfs_multi",
        profile="sampling",
        extension_evaluator=extension_evaluator,
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
    self.assertFalse(called["fallback"])
    self.assertTrue(called["extension"])
    self.assertEqual(out["score"], 0.8)

  def test_sampling_profile_filters_unsupported_extension_kwargs(self):
    called = {"fallback": False, "extension": False}
    seen = {}

    def extension_evaluator(
        *,
        sampler,
        predict_fn,
        sample_count,
        rng_key,
        extras,
        save_results_fn,
        filename,
        NSE,
    ):
      del sampler, predict_fn, sample_count, rng_key, extras, save_results_fn
      seen["filename"] = filename
      seen["NSE"] = NSE
      called["extension"] = True
      return {"score": 0.7}

    def fallback_eval_fn(**kwargs):
      del kwargs
      called["fallback"] = True
      return {"score": 0.1}

    out = dispatch.evaluate_with_optional_extension(
        algorithm_name="dfs_multi",
        profile="sampling",
        extension_evaluator=extension_evaluator,
        sampler=object(),
        predict_fn=object(),
        sample_count=3,
        rng_key=0,
        extras={},
        artifact_prefix="unused",
        save_artifacts=False,
        fallback_eval_fn=fallback_eval_fn,
        extension_kwargs={"NSE": 17, "vd_flag": True},
    )
    self.assertFalse(called["fallback"])
    self.assertTrue(called["extension"])
    self.assertEqual(out["score"], 0.7)
    self.assertEqual(seen["NSE"], 17)
    self.assertEqual(seen["filename"], "unused_dfs_multi")

  def test_registry_dispatch_uses_registered_extension(self):
    called = {"fallback": False, "extension": False}
    seen = {}

    class _Extension:
      def __init__(self, evaluator):
        self.evaluator = evaluator

    class _Registry:
      def get_extension(self, algorithm_name):
        if algorithm_name == "dfs_multi":
          return _Extension(extension_evaluator)
        return None

    def extension_evaluator(**kwargs):
      called["extension"] = True
      seen["filename"] = kwargs["filename"]
      return {"score": 0.9}

    def fallback_eval_fn(**kwargs):
      del kwargs
      called["fallback"] = True
      return {"score": 0.1}

    original_registry = dispatch._MULTISOL_REGISTRY
    dispatch._MULTISOL_REGISTRY = _Registry()
    try:
      out = dispatch.evaluate_with_registry(
          algorithm_name="dfs_multi",
          split="test",
          profile="sampling",
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
      dispatch._MULTISOL_REGISTRY = original_registry

    self.assertFalse(called["fallback"])
    self.assertTrue(called["extension"])
    self.assertEqual(seen["filename"], "sampling_eval_test_dfs_multi")
    self.assertEqual(out["score"], 0.9)

  def test_registry_dispatch_falls_back_for_non_extension_algorithm(self):
    called = {"fallback": False}
    seen = {}
    fallback_kwargs = {}

    class _Registry:
      def get_extension(self, algorithm_name):
        seen["algorithm_name"] = algorithm_name
        return None

    sampler = object()
    predict_fn = object()
    extras = {"examples_seen": 7}

    def fallback_eval_fn(**kwargs):
      called["fallback"] = True
      fallback_kwargs.update(kwargs)
      return {"score": 0.5}

    original_registry = dispatch._MULTISOL_REGISTRY
    dispatch._MULTISOL_REGISTRY = _Registry()
    try:
      out = dispatch.evaluate_with_registry(
          algorithm_name="bfs",
          split="val",
          profile="sampling",
          sampler=sampler,
          predict_fn=predict_fn,
          sample_count=3,
          rng_key=0,
          extras=extras,
          artifact_prefix="sampling_eval",
          save_artifacts=False,
          fallback_eval_fn=fallback_eval_fn,
      )
    finally:
      dispatch._MULTISOL_REGISTRY = original_registry

    self.assertTrue(called["fallback"])
    self.assertEqual(seen["algorithm_name"], "bfs")
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

  def test_registry_dispatch_uses_upstream_path_for_default_profile(self):
    called = {"fallback": False}

    class _Registry:
      def get_extension(self, algorithm_name):
        raise AssertionError(
            f"Registry lookup should not run for default profile: {algorithm_name}"
        )

    def fallback_eval_fn(**kwargs):
      del kwargs
      called["fallback"] = True
      return {"score": 0.4}

    original_registry = dispatch._MULTISOL_REGISTRY
    dispatch._MULTISOL_REGISTRY = _Registry()
    try:
      out = dispatch.evaluate_with_registry(
          algorithm_name="dfs_multi",
          split="val",
          profile="default",
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
      dispatch._MULTISOL_REGISTRY = original_registry

    self.assertTrue(called["fallback"])
    self.assertEqual(out["score"], 0.4)

  def test_run_flag_defaults_preserve_upstream_behavior(self):
    defaults = self._run_flag_defaults()
    self.assertEqual(defaults["evaluation_profile"], "default")
    self.assertIs(defaults["save_sampling_artifacts"], False)
    self.assertEqual(defaults["sampling_artifact_prefix"], "sampling_eval")
    self.assertEqual(defaults["run_dir"], "")
    self.assertEqual(defaults["filename"], "")
    self.assertIs(defaults["results_df"], False)
    self.assertIs(defaults["save_df"], False)
    self.assertIs(defaults["save_model_to_file"], False)
    self.assertIs(defaults["validate_distributions"], False)
    self.assertEqual(defaults["NSE"], 25)
    self.assertIsNone(defaults["test_length"])

  def test_run_source_resolves_checkpoint_default_under_run_dir(self):
    run_path = pathlib.Path(__file__).resolve().with_name("run.py")
    source = run_path.read_text(encoding="utf-8")
    self.assertIn("def _resolve_checkpoint_path", source)
    self.assertIn("checkpoint_flag = FLAGS['checkpoint_path']", source)
    self.assertIn("if checkpoint_flag.present", source)
    self.assertIn("os.path.join(run_dir, 'checkpoints')", source)


if __name__ == "__main__":
  absltest.main()
