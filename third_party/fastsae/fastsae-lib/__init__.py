"""
SAE (Sparse Autoencoder) Package

A package for efficient training of Sparse Autoencoders.
"""

__version__ = "0.1.0"

# Re-export common subpackages at the top level so attribute access works:
#   import fastsae; fastsae.models / fastsae.dataset / fastsae.training / fastsae.interpret
# This keeps backward-compat with examples that expect attribute-style access.
from . import dataset as dataset  # noqa: F401
from . import interpret as interpret  # noqa: F401
from . import models as models  # noqa: F401
from . import sklearn as sklearn  # noqa: F401
from . import training as training  # noqa: F401

__all__ = [
    "models",
    "dataset",
    "training",
    "interpret",
    "sklearn",
    "__version__",
]
