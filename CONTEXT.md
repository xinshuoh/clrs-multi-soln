# Context

## Domain Vocabulary

**Multi-Solution Algorithm Definition**
The single source of truth for a CLRS multi-solution extension. It names the
algorithm, its base CLRS algorithm, its spec, its training data generator, its
sampler, its stochastic extraction methods, and its validation method.

**Multi-Solution Algorithm**
The author-facing Python object that represents one multi-solution algorithm in
the framework. It owns the spec, Training Distribution, randomized symbolic
algorithm, Solution Space, sampler, and CLRS training entry point.

**Randomized Algorithm**
The randomized symbolic algorithm that can be run once to produce one concrete
solution, or repeatedly to build a Training Distribution.

**Built-in Definition Catalog**
The explicit Python list of multi-solution algorithm definitions shipped inside
the repository. Registering a new built-in algorithm means adding its definition
object to this catalog.

**Training Distribution**
The empirical target distribution produced by repeatedly running a randomized
symbolic algorithm on one input instance and aggregating the sampled solutions.
It includes the number of randomized symbolic executions and the output encoding
used to turn sampled concrete solutions into a model target.

**Extraction Method**
A stochastic method that samples a concrete solution from either the model's
predicted distribution or the empirical target distribution.

**Validation Method**
The algorithm-specific check that decides whether a sampled concrete solution
could have been produced by the randomized symbolic generator for the same
input.
