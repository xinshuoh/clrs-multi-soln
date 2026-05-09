"""Tests for configurable report artifact sinks."""

import pathlib
import pickle
import tempfile

from absl.testing import absltest

from clrs._src.multi_sol.evaluation import reporting


class ReportingTest(absltest.TestCase):

  def test_save_csv_report_without_timestamp_uses_exact_name(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      reporting.save_csv_report(
          {"metric": [1.0]},
          "legacy_eval",
          output_dir=tmpdir,
          timestamped=False,
      )
      output_path = pathlib.Path(tmpdir) / "legacy_eval.csv"
      self.assertTrue(output_path.exists())

  def test_save_pickle_report_without_timestamp_uses_exact_name(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      payload = {"metric": [1.0]}
      reporting.save_pickle_report(
          payload,
          "legacy_eval",
          output_dir=tmpdir,
          timestamped=False,
      )
      output_path = pathlib.Path(tmpdir) / "legacy_eval.pkl"
      self.assertTrue(output_path.exists())
      with open(output_path, "rb") as f:
        restored = pickle.load(f)
      self.assertEqual(restored, payload)


if __name__ == "__main__":
  absltest.main()
