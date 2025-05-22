import pluggy

hookimpl = pluggy.HookimplMarker("insilicoICH")
"""Marker to be imported and used in plugins (and for own implementations)"""

from . import image_acquisition
from . import lesion_definition
from . import phantoms
from . import artifact_generation
from . import study
from . import hooks
from .study import load_phantom, available_phantoms
from .image_acquisition import Scanner
