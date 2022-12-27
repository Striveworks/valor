import importlib.metadata

try:
    __version__ = importlib.metadata.version("functional-cat")
except importlib.metadata.PackageNotFoundError:
    __version__ = ""
