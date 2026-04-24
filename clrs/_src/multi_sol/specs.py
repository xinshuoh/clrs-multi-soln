"""Multi-solution algorithm specs keyed by extension algorithm name."""

from __future__ import annotations

import types

from clrs._src import specs


MULTI_SOL_SPECS = types.MappingProxyType({
    "dfs_multi": {
        "pos": (specs.Stage.INPUT, specs.Location.NODE, specs.Type.SCALAR),
        "A": (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.SCALAR),
        "adj": (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        "pi": (
            specs.Stage.OUTPUT,
            specs.Location.NODE,
            specs.Type.POINTER_DISTRIBUTION,
        ),
        "pi_h": (specs.Stage.HINT, specs.Location.NODE, specs.Type.POINTER),
        "color": (specs.Stage.HINT, specs.Location.NODE, specs.Type.CATEGORICAL),
        "d": (specs.Stage.HINT, specs.Location.NODE, specs.Type.SCALAR),
        "f": (specs.Stage.HINT, specs.Location.NODE, specs.Type.SCALAR),
        "s_prev": (specs.Stage.HINT, specs.Location.NODE, specs.Type.POINTER),
        "s": (specs.Stage.HINT, specs.Location.NODE, specs.Type.MASK_ONE),
        "u": (specs.Stage.HINT, specs.Location.NODE, specs.Type.MASK_ONE),
        "v": (specs.Stage.HINT, specs.Location.NODE, specs.Type.MASK_ONE),
        "s_last": (specs.Stage.HINT, specs.Location.NODE, specs.Type.MASK_ONE),
        "time": (specs.Stage.HINT, specs.Location.GRAPH, specs.Type.SCALAR),
    },
    "bfs_multi": {
        "pos": (specs.Stage.INPUT, specs.Location.NODE, specs.Type.SCALAR),
        "s": (specs.Stage.INPUT, specs.Location.NODE, specs.Type.MASK_ONE),
        "A": (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.SCALAR),
        "adj": (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        "pi": (
            specs.Stage.OUTPUT,
            specs.Location.NODE,
            specs.Type.POINTER_DISTRIBUTION,
        ),
        "reach_h": (specs.Stage.HINT, specs.Location.NODE, specs.Type.MASK),
        "pi_h": (specs.Stage.HINT, specs.Location.NODE, specs.Type.POINTER),
    },
    "bellman_ford_multi": {
        "pos": (specs.Stage.INPUT, specs.Location.NODE, specs.Type.SCALAR),
        "s": (specs.Stage.INPUT, specs.Location.NODE, specs.Type.MASK_ONE),
        "A": (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.SCALAR),
        "adj": (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        "pi": (
            specs.Stage.OUTPUT,
            specs.Location.NODE,
            specs.Type.POINTER_DISTRIBUTION,
        ),
        "pi_h": (specs.Stage.HINT, specs.Location.NODE, specs.Type.POINTER),
        "d": (specs.Stage.HINT, specs.Location.NODE, specs.Type.SCALAR),
        "msk": (specs.Stage.HINT, specs.Location.NODE, specs.Type.MASK),
    },
})

