from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("spec2vec")
except PackageNotFoundError:
    __version__ = "0+unknown"
