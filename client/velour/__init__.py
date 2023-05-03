import sys

from .client import Client, Dataset

__all__ = ["Client", "Dataset"]

if sys.version_info.minor >= 8:
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata

try:
    __version__ = importlib_metadata.version("velour")
except importlib_metadata.PackageNotFoundError:
    __version__ = ""
