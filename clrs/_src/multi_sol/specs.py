"""Multi-solution algorithm specs keyed by extension algorithm name."""

from __future__ import annotations

import types

from clrs._src.specs import Location, Stage, Type

MULTI_SOL_SPECS = types.MappingProxyType({
    "dfs_multi": {
        "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
        "A": (Stage.INPUT, Location.EDGE, Type.SCALAR),
        "adj": (Stage.INPUT, Location.EDGE, Type.MASK),
        "pi": (
            Stage.OUTPUT,
            Location.NODE,
            Type.POINTER_DISTRIBUTION,
        ),
        "pi_h": (Stage.HINT, Location.NODE, Type.POINTER),
        "color": (Stage.HINT, Location.NODE, Type.CATEGORICAL),
        "d": (Stage.HINT, Location.NODE, Type.SCALAR),
        "f": (Stage.HINT, Location.NODE, Type.SCALAR),
        "s_prev": (Stage.HINT, Location.NODE, Type.POINTER),
        "s": (Stage.HINT, Location.NODE, Type.MASK_ONE),
        "u": (Stage.HINT, Location.NODE, Type.MASK_ONE),
        "v": (Stage.HINT, Location.NODE, Type.MASK_ONE),
        "s_last": (Stage.HINT, Location.NODE, Type.MASK_ONE),
        "time": (Stage.HINT, Location.GRAPH, Type.SCALAR),
    },
    "bfs_multi": {
        "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
        "s": (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        "A": (Stage.INPUT, Location.EDGE, Type.SCALAR),
        "adj": (Stage.INPUT, Location.EDGE, Type.MASK),
        "pi": (
            Stage.OUTPUT,
            Location.NODE,
            Type.POINTER_DISTRIBUTION,
        ),
        "reach_h": (Stage.HINT, Location.NODE, Type.MASK),
        "pi_h": (Stage.HINT, Location.NODE, Type.POINTER),
    },
    "bellman_ford_multi": {
        "pos": (Stage.INPUT, Location.NODE, Type.SCALAR),
        "s": (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        "A": (Stage.INPUT, Location.EDGE, Type.SCALAR),
        "adj": (Stage.INPUT, Location.EDGE, Type.MASK),
        "pi": (
            Stage.OUTPUT,
            Location.NODE,
            Type.POINTER_DISTRIBUTION,
        ),
        "pi_h": (Stage.HINT, Location.NODE, Type.POINTER),
        "d": (Stage.HINT, Location.NODE, Type.SCALAR),
        "msk": (Stage.HINT, Location.NODE, Type.MASK),
    },
    "mst_prim_multi": {
        'pos': (Stage.INPUT, Location.NODE, Type.SCALAR),
        's': (Stage.INPUT, Location.NODE, Type.MASK_ONE),
        'A': (Stage.INPUT, Location.EDGE, Type.SCALAR),
        'adj': (Stage.INPUT, Location.EDGE, Type.MASK),
        'pi': (Stage.OUTPUT, Location.NODE, Type.POINTER_DISTRIBUTION),
        'pi_h': (Stage.HINT, Location.NODE, Type.POINTER),
        'key': (Stage.HINT, Location.NODE, Type.SCALAR),
        'mark': (Stage.HINT, Location.NODE, Type.MASK),
        'in_queue': (Stage.HINT, Location.NODE, Type.MASK),
        'u': (Stage.HINT, Location.NODE, Type.MASK_ONE)
    },
})

