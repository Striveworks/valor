import sys

if sys.version_info.minor >= 8:
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata

try:
    __version__ = importlib_metadata.version("velour")
except importlib_metadata.PackageNotFoundError:
    __version__ = ""

from .client import (
    Client, 
    ClientException,
    Dataset,
    Evaluation,
    GroundTruth,
    Model,
    Prediction,
)
from .schemas.core import (
    Annotation,
    Datum,
    Label,
)

__all__ = [
    "Client",
    "ClientException",
    "Label",
    "Evaluation",
    "Dataset",
    "Model",
    "Datum",
    "Annotation",
    "GroundTruth",
    "Prediction",
]
