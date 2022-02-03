from .__version__ import __version__
from .Document import Document
from .logging_functions import _init_logger
from .Spec2Vec import Spec2Vec
from .SpectrumDocument import SpectrumDocument
from .vector_operations import calc_vector


_init_logger()


__all__ = [
    "__version__",
    "calc_vector",
    "Document",
    "SpectrumDocument",
    "Spec2Vec",
]
