"""Compatibility view of multi-solution algorithm specs."""

from __future__ import annotations

import types

from clrs._src.multi_sol.algorithms import builtins


MULTI_SOL_SPECS = types.MappingProxyType({
    definition.algorithm_name: definition.spec
    for definition in builtins.BUILTIN_DEFINITIONS
})
