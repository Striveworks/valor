import sys

if sys.version_info.minor >= 8:
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata  # type: ignore

try:
    __version__ = importlib_metadata.version("velour")
except importlib_metadata.PackageNotFoundError:
    __version__ = ""

from .client import ClientConnection
from .coretypes import (
    Annotation,
    Client,
    Dataset,
    Datum,
    Evaluation,
    GroundTruth,
    Label,
    Model,
    Prediction,
)
from .schemas import Filter

__all__ = [
    "Client",
    "ClientConnection",
    "Label",
    "Evaluation",
    "Dataset",
    "Model",
    "Datum",
    "Annotation",
    "GroundTruth",
    "Prediction",
    "Filter",
]
