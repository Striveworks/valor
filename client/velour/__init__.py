import sys

if sys.version_info.minor >= 8:
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata

try:
    __version__ = importlib_metadata.version("velour")
except importlib_metadata.PackageNotFoundError:
    __version__ = ""

from .client import Client, ClientException, Dataset, Evaluation
from .coretypes import Annotation, Datum, GroundTruth, Label, Prediction
from .metatypes import ImageMetadata, VideoFrameMetadata

__all__ = [
    "Client",
    "ClientException",
    "Evaluation",
    "Dataset",
    "Model",
    "Datum",
    "Annotation",
    "GroundTruth",
    "Prediction",
    "Label",
    "ImageMetadata",
    "VideoFrameMetadata",
]
