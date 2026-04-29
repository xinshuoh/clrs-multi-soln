# Context

## Domain Vocabulary

**Multi-Solution Algorithm Definition**
The single source of truth for a CLRS multi-solution extension. It names the
algorithm, its base CLRS algorithm, its spec, its training data generator, its
sampler, its stochastic extraction methods, and its validation method.

**Built-in Definition Catalog**
The explicit Python list of multi-solution algorithm definitions shipped inside
the repository. Registering a new built-in algorithm means adding its definition
object to this catalog.

**Training Distribution**
The empirical target distribution produced by repeatedly running a randomized
symbolic algorithm on one input instance and aggregating the sampled solutions.

**Extraction Method**
A stochastic method that samples a concrete solution from either the model's
predicted distribution or the empirical target distribution.

**Validation Method**
The algorithm-specific check that decides whether a sampled concrete solution
could have been produced by the randomized symbolic generator for the same
input.
