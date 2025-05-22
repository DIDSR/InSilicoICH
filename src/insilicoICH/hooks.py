# hooks.py

import pluggy
from typing import Dict  # For type hinting
from .phantoms.base_phantoms import Phantom

PROJECT_NAME = "insilicoich"  # Choose a unique name for your plugin system
hookspec = pluggy.HookspecMarker(PROJECT_NAME)
hookimpl = pluggy.HookimplMarker(PROJECT_NAME)


class PhantomSpecs:
    """Hook specifications for phantom plugins."""

    @hookspec
    def register_phantom_types(self) -> Dict[str, Phantom]:  # type: ignore
        """
        Plugins implement this hook to register their Phantom subclasses.

        Each implementation returns a dict of Phantom subclasses they provide.
        The main application will collect these dicts.
        Return an empty dict or None if a plugin has no types to register.
        """
        return {}  # Default implementation returns an empty list
