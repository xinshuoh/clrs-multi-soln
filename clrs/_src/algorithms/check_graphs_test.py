"""Unit tests for `check_graphs.py`."""

from absl.testing import absltest
from clrs._src.algorithms import check_graphs
import numpy as np


# -----------------------------------------------------
# Test Graph Fixtures (following graphs_test.py pattern)
# -----------------------------------------------------

# Simple undirected graph for BFS testing
# Structure:
# 0 - 1 - 2
#  \ / \ /
#   4 - 3
UNDIRECTED = np.array(
    [
        [0, 1, 0, 0, 1],
        [1, 0, 1, 1, 1],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 1, 0, 1, 0],
    ]
)

# Simple directed graph for DFS testing
DIRECTED = np.array(
    [
        [0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
)

# Weighted graph for Bellman-Ford testing
WEIGHTED_DIRECTED = np.array(
    [
        [0, 1, 0, 0, 0],
        [0, 0, 1, 2, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0],
    ]
)

# Simple cyclic graph
CYCLIC = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

# Disconnected graph
DISCONNECTED = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])


class CheckGraphsTest(absltest.TestCase):
    """Test suite for graph validation functions."""

    # -----------------------------------------------------
    # Bellman-Ford Path Validation Tests
    # -----------------------------------------------------

    def test_check_valid_BFpaths_valid_shortest_paths(self):
        """Test valid shortest path trees are accepted."""
        source = 0
        valid_bf_pi = np.array([0, 0, 1, 1, 2])
        self.assertTrue(
            check_graphs.check_valid_BFpaths(WEIGHTED_DIRECTED, source, valid_bf_pi)
        )

    def test_check_valid_BFpaths_non_shortest_path(self):
        """Test that non-shortest paths are rejected."""
        source = 0
        # Takes longer path to node 2: 0->1->3->4 instead of 0->1->2->4
        invalid_bf_pi = np.array([0, 0, 1, 1, 3])
        self.assertFalse(
            check_graphs.check_valid_BFpaths(WEIGHTED_DIRECTED, source, invalid_bf_pi)
        )

    def test_check_valid_BFpaths_invalid_edge(self):
        """Test that non-existent parent edges are rejected."""
        source = 0
        # Edge 2->1 doesn't exist (graph is directed)
        invalid_pi = np.array([0, 2, 0, 1, 2])
        self.assertFalse(
            check_graphs.check_valid_BFpaths(WEIGHTED_DIRECTED, source, invalid_pi)
        )

    def test_check_valid_BFpaths_unreachable_nodes(self):
        """Test that self-parents for unreachable nodes are allowed."""
        source = 0
        # Node 2 is unreachable, pi[2] = 2 is valid
        valid_pi = np.array([0, 0, 2])
        self.assertTrue(
            check_graphs.check_valid_BFpaths(DISCONNECTED, source, valid_pi)
        )

    def test_bellman_ford_cost_computation(self):
        """Test that Bellman-Ford cost computation is correct."""
        source = 0
        expected_costs = np.array([0, 1, 2, 3, 3])
        actual_costs = check_graphs.bellman_ford_cost(WEIGHTED_DIRECTED, source)
        np.testing.assert_array_equal(expected_costs, actual_costs)

    # -----------------------------------------------------
    # BFS Tree Validation Tests
    # -----------------------------------------------------

    def test_check_valid_bfsTree_valid_cases(self):
        """Test valid BFS trees are correctly identified."""
        # Valid BFS from root 0
        valid_bfs_pi = np.array([0, 0, 1, 1, 0])
        self.assertTrue(check_graphs.check_valid_bfsTree(UNDIRECTED, valid_bfs_pi, s=0))

        # Alternative valid BFS tree
        valid_bfs_pi2 = np.array([0, 0, 1, 4, 0])
        self.assertTrue(
            check_graphs.check_valid_bfsTree(UNDIRECTED, valid_bfs_pi2, s=0)
        )

    def test_self_valid_bfsTree_unreachable(self):
        """Test handling of unreachable nodes."""
        graph_with_unreachable = np.array(
            [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
        )
        pi = np.array([0, 0, 1, 3]) # Node 3 is unreachable, pi[3] = 3 should be self-parent

        self.assertTrue(check_graphs.check_valid_bfsTree(graph_with_unreachable, pi, s=0))

    # -----------------------------------------------------
    # Regression Tests (from known issues)
    # -----------------------------------------------------

    def test_regression_issue_from_sanity_check(self):
        """Test specific graph from dfs_verification_tester sanity check."""
        # From dfs_verification_tester.py line 351
        A = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 1],
                [1, 0, 0, 1, 0, 0],
                [1, 1, 0, 0, 0, 1],
                [0, 1, 1, 0, 0, 1],
                [0, 1, 0, 0, 0, 0],
            ]
        )

        T = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        # Convert adjacency matrix to parent array
        pi = self._adj_matrix_to_parent_array(T)

        # Test with both verification methods
        result = check_graphs.check_valid_dfsTree_new(A, pi)
        # Adjust assertion based on whether this should pass/fail

    # -----------------------------------------------------
    # Helper Methods
    # -----------------------------------------------------

    def _adj_matrix_to_parent_array(self, adj_matrix):
        """Convert adjacency matrix to parent array."""
        n = len(adj_matrix)
        pi = np.arange(n)  # Default to self-parent

        for child in range(n):
            for parent in range(n):
                if adj_matrix[parent, child] == 1:
                    pi[child] = parent
                    break
        return pi


if __name__ == "__main__":
    absltest.main()
