"""
Functions for exporting and importing trained :class:`~gensim.models.Word2Vec` model to and from disk.
##########################################
Functions provide the ability to export and import trained :class:`~gensim.models.Word2Vec` model to and from disk
without pickling the model. The model can be stored in two files: `.json` for metadata and `.npy` for weights.
"""
from .model_exporting import export_model
from .model_importing import Word2VecLight, import_model


__all__ = [
    "export_model",
    "import_model",
    "Word2VecLight"
    ]
