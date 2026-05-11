# Context

## Domain Vocabulary

**Multi-Solution Algorithm Definition**
The single source of truth for a CLRS multi-solution extension.

**Multi-Solution Algorithm**
The author-facing Python object that represents one multi-solution algorithm in
the framework. 

**Extraction Method**
A stochastic method that samples a concrete solution from either the model's
predicted distribution or the empirical target distribution.

**Validation Method**
The algorithm-specific check that decides whether a sampled concrete solution
could have been produced by the randomized symbolic generator for the same
input.
