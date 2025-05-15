from .phantoms.base_phantoms import Phantom

# hooks.py
import pluggy
from typing import List  # For type hinting

# It's good practice to import the base class if the hook spec refers to it,
# even if just for type hinting or context.
# from .base_phantom import Phantom # Assuming Phantom is in base_phantom.py

PROJECT_NAME = "insilicoich"  # Choose a unique name for your plugin system
hookspec = pluggy.HookspecMarker(PROJECT_NAME)
hookimpl = pluggy.HookimplMarker(PROJECT_NAME)


class PhantomSpecs:
    """Hook specifications for phantom plugins."""

    @hookspec
    def register_phantom_types(self) -> List[Phantom]:  # type: ignore
        """
        Plugins implement this hook to register their Phantom subclasses.

        Each implementation returns a list of Phantom subclasses they provide.
        The main application will collect these lists.
        Return an empty list or None if a plugin has no types to register.
        """
        return []  # Default implementation returns an empty list
