import json
import os
from typing import Union
import numpy as np
import scipy.sparse
from gensim.models import KeyedVectors


class Word2VecLight:
    """
    A lightweight version of :class:`~gensim.models.Word2Vec`. The objects of this class follow the interface of the
    original :class:`~gensim.models.Word2Vec` to the point necessary to calculate Spec2Vec scores. The model cannot be
    used for further training.
    """

    def __init__(self, model: dict, weights: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
        """

        Parameters
        ----------
        model:
            A dictionary containing the model's metadata.
        weights:
            A numpy array or a scipy sparse matrix containing the model's weights.
        """
        self.wv: KeyedVectors = self._KeyedVectorsBuilder().from_dict(model).with_weights(weights).build()

    class _KeyedVectorsBuilder:
        def __init__(self):
            self.vector_size = None
            self.weights = None

        def build(self) -> KeyedVectors:
            keyed_vectors = KeyedVectors(self.vector_size)
            keyed_vectors.__dict__ = self.__dict__
            keyed_vectors.vectors = self.weights
            return keyed_vectors

        def from_dict(self, dictionary: dict):
            expected_keys = {"vector_size", "__numpys", "__scipys", "__ignoreds", "__recursive_saveloads",
                             "index_to_key", "norms", "key_to_index", "next_index", "__weights_format"}
            if dictionary.keys() == expected_keys:
                self.__dict__ = dictionary
            else:
                raise ValueError("The keys of model's dictionary representation do not match the expected keys.")
            return self

        def with_weights(self, weights: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
            self.weights = weights
            return self


def import_model(model_file, weights_file) -> Word2VecLight:
    """
    Read a lightweight version of a :class:`~gensim.models.Word2Vec` model from disk.

    Parameters
    ----------
    model_file:
        A path of json file to load the model.
    weights_file:
        A path of `.npy` file to load the model's weights.

    Returns
    -------
    :class:`~spec2vec.serialization.model_importing.Word2VecLight` – a lightweight version of a
    :class:`~gensim.models.Word2Vec`
    """
    with open(model_file, "r", encoding="utf-8") as f:
        model: dict = json.load(f)

    weights = load_weights(weights_file, model["__weights_format"])
    return Word2VecLight(model, weights)


def load_weights(weights_file: Union[str, os.PathLike],
                 weights_format: str) -> Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]:
    weights: np.ndarray = np.load(weights_file, allow_pickle=False)

    weights_array_builder = {"csr_matrix": scipy.sparse.csr_matrix,
                            "csc_matrix": scipy.sparse.csc_matrix,
                            "np.ndarray": lambda x: x}
    weights = weights_array_builder[weights_format](weights)

    return weights
