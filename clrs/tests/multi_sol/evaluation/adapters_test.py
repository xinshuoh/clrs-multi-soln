"""Tests for multi-solution CLRS feedback adapters."""

from absl.testing import absltest

import numpy as np

from clrs._src.multi_sol.evaluation import adapters


class DummyDataPoint:

  def __init__(self, data):
    self.data = np.asarray(data)


class DummyFeedback:

  def __init__(self, inputs):
    self._inputs = inputs

  def __getitem__(self, idx):
    if idx != 0:
      raise IndexError(idx)
    return [self._inputs]


class FeedbackAdaptersTest(absltest.TestCase):

  def test_extract_dfs_graph_and_source(self):
    adjacency = np.asarray([
        [[0, 1], [0, 0]],
        [[0, 0], [1, 0]],
    ])
    feedback = DummyFeedback([
        DummyDataPoint(np.zeros((2, 2))),
        DummyDataPoint(adjacency),
    ])

    extracted_adjacency, source_nodes = adapters.extract_dfs_graph_and_source(
        feedback)

    np.testing.assert_array_equal(extracted_adjacency, adjacency)
    np.testing.assert_array_equal(source_nodes, np.asarray([0, 0]))

  def test_extract_bfs_graph_and_source(self):
    adjacency = np.asarray([[[0, 1], [1, 0]]])
    source_one_hot = np.asarray([[0, 1]])
    feedback = _source_graph_feedback(source_one_hot, adjacency)

    extracted_adjacency, source_nodes = adapters.extract_bfs_graph_and_source(
        feedback)

    np.testing.assert_array_equal(extracted_adjacency, adjacency)
    np.testing.assert_array_equal(source_nodes, np.asarray([1]))

  def test_extract_bellman_ford_graph_and_source(self):
    adjacency = np.asarray([[[0, 3], [0, 0]]])
    source_one_hot = np.asarray([[1, 0]])
    feedback = _source_graph_feedback(source_one_hot, adjacency)

    extracted_adjacency, source_nodes = (
        adapters.extract_bellman_ford_graph_and_source(feedback))

    np.testing.assert_array_equal(extracted_adjacency, adjacency)
    np.testing.assert_array_equal(source_nodes, np.asarray([0]))

  def test_extract_mst_prim_graph_and_source(self):
    adjacency = np.asarray([[[0, 2], [2, 0]]])
    source_one_hot = np.asarray([[0, 1]])
    feedback = _source_graph_feedback(source_one_hot, adjacency)

    extracted_adjacency, source_nodes = (
        adapters.extract_mst_prim_graph_and_source(feedback))

    np.testing.assert_array_equal(extracted_adjacency, adjacency)
    np.testing.assert_array_equal(source_nodes, np.asarray([1]))


def _source_graph_feedback(source_one_hot, adjacency):
  return DummyFeedback([
      DummyDataPoint(np.zeros((len(adjacency), adjacency.shape[-1]))),
      DummyDataPoint(source_one_hot),
      DummyDataPoint(adjacency),
  ])


if __name__ == "__main__":
  absltest.main()
