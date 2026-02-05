import numpy as np
import unittest
from clrs._src import bfs_sampling

class DummyDatapoint:
    def __init__(self, data):
        self.data = data

class BFSSamplingTest(unittest.TestCase):
    def test_prim_like_sampler_simple_path(self):
        # Graph: 0 -> 1 -> 2
        # Matrix index: [row][col] = prob that col is parent of row
        # 0 is root.
        # 1's parent should be 0.
        # 2's parent should be 1.
        
        #       0    1    2
        # 0 [1.0, 0.0, 0.0] (0 points to 0)
        # 1 [1.0, 0.0, 0.0] (1 points to 0)
        # 2 [0.0, 1.0, 0.0] (2 points to 1)
        
        probMatrix = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        
        # Wrap in dummy structure for extraction
        # The extraction expects a list of datapoints, where each datapoint.data is a list of probMatrices
        outsOrPreds = [DummyDatapoint([probMatrix])]
        s_indices = [0]
        
        trees = bfs_sampling.sample_bfs_prim(outsOrPreds, s_indices)
        
        # Expected tree: pi[0]=0, pi[1]=0, pi[2]=1
        expected_pi = np.array([0, 0, 1])
        
        np.testing.assert_array_equal(trees[0], expected_pi)

    def test_prim_like_sampler_branching(self):
        # Graph: 0 -> 1, 0 -> 2
        # 1 parent 0
        # 2 parent 0
        
        probMatrix = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        
        outsOrPreds = [DummyDatapoint([probMatrix])]
        s_indices = [0]
        
        trees = bfs_sampling.sample_bfs_prim(outsOrPreds, s_indices)
        expected_pi = np.array([0, 0, 0])
        np.testing.assert_array_equal(trees[0], expected_pi)
        
    def test_prim_like_sampler_disconnected(self):
        # Graph: 0 (root), 1 (disconnected/self), 2 (disconnected/self)
        # Although probMatrix might imply something, if there's NO probability from processed set...
        
        # Let's say 0 is root.
        # 1 has only probability to 2 (unreachable from 0 initially)
        # 2 has only probability to 1
        
        # This is a disconnected graph case. The sampler should handle it gracefully.
        
        probMatrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0], # 1 points to 2
            [0.0, 1.0, 0.0]  # 2 points to 1
        ])
        
        outsOrPreds = [DummyDatapoint([probMatrix])]
        s_indices = [0]
        
        # This might be non-deterministic if we don't seed, but logic says:
        # processed = {0}
        # remaining = {1, 2}
        # score(1) from processed(0) = 0
        # score(2) from processed(0) = 0
        # It picks one. Say 1. Parent(1) from processed? None valid. 
        # Implementation: defaults to self if no valid parent.
        # So pi[1] -> 1. processed={0,1}.
        # remaining={2}.
        # score(2) from processed{0,1} -> prob(2->0)=0 + prob(2->1)=1 = 1.
        # So 2 connects to 1.
        # Result: 0->0, 1->1, 2->1.
        
        # Or if 2 picked first: 2->2, 1->2.
        
        # Let's see if it crashes.
        trees = bfs_sampling.sample_bfs_prim(outsOrPreds, s_indices)
        self.assertEqual(len(trees), 1)
        self.assertEqual(len(trees[0]), 3)
        self.assertEqual(trees[0][0], 0) # Root always itself

    def test_beam_search(self):
        # Graph: 0->1, 0->2. 
        # Probabilities favor 0->1 over 0->2 slightly for 1.
        
        probMatrix = np.array([
            [1.0, 0.0, 0.0],
            [0.6, 0.4, 0.0], # 1 prefers 0(0.6) over 1(0.4/self??) -- wait, probMatrix is P(parent(i)=j)
                             # Let's say P(parent(1)=0)=0.9, P(parent(1)=2)=0.1
            [0.9, 0.0, 0.1]  # P(parent(2)=0)=0.9
        ])
        
        outsOrPreds = [DummyDatapoint([probMatrix])]
        s_indices = [0]
        
        # Greedy might just pick max at each step. 
        # Beam search with width 1 should match greedy roughly (if no lookahead issue).
        
        trees = bfs_sampling.sample_bfs_beam(outsOrPreds, s_indices, beam_width=2)
        # Should pick 0->1, 0->2.
        expected_pi = np.array([0, 0, 0])
        np.testing.assert_array_equal(trees[0], expected_pi)

if __name__ == '__main__':
    unittest.main()
