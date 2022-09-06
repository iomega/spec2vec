from gensim.models import KeyedVectors
import json
import numpy as np
import scipy.sparse
from typing import Union


class Word2VecLight:
    def __init__(self, model: dict, weights: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
        self.wv = self._KeyedVectorsBuilder().from_dict(model).with_weights(weights).build()

    class _KeyedVectorsBuilder:
        def __init__(self):
            self.weights = None
            self.key_to_index = None
            self.index_to_key = None
            self.vector_size = None

        def build(self) -> KeyedVectors:
            keyed_vectors = KeyedVectors(self.vector_size)
            keyed_vectors.vector_size = self.vector_size
            keyed_vectors.index_to_key = self.index_to_key
            keyed_vectors.key_to_index = self.key_to_index
            keyed_vectors.vectors = self.weights
            return keyed_vectors

        def from_dict(self, dictionary: dict):
            self.vector_size = dictionary["vector_size"]
            self.index_to_key = dictionary["index_to_key"]
            self.key_to_index = dictionary["key_to_index"]
            return self

        def with_weights(self, weights: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
            self.weights = weights
            return self


def import_model(model_file, weights_file) -> Word2VecLight:
    with open(model_file, "r") as f:
        model: dict = json.load(f)

    if model["__numpys"] or model["__ignoreds"]:
        weights = np.load(weights_file)
    elif model["__scipys"]:
        weights = scipy.sparse.load_npz(weights_file)
    else:
        raise ValueError("The model's weights format is undefined.")

    return Word2VecLight(model, weights)
