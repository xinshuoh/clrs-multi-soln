"""Built-in multi-solution extension registrations from catalog."""

from clrs._src.multi_sol import catalog
from clrs._src.multi_sol.core import registry


def _register_builtin_extensions() -> None:
  for extension in catalog.BUILTIN_EXTENSIONS:
    try:
      registry.register_extension(extension)
    except ValueError:
      continue


_register_builtin_extensions()
