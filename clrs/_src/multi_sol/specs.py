"""Compatibility view of multi-solution algorithm specs."""

from __future__ import annotations

import types

from clrs._src.multi_sol import catalog


MULTI_SOL_SPECS = types.MappingProxyType({
    definition.algorithm_name: definition.spec
    for definition in catalog.BUILTIN_DEFINITIONS
})
