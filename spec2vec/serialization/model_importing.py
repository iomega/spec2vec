import json
import numpy as np
import scipy.sparse
from typing import Union


class Word2VecLight:
    def __init__(self, model: dict, weights: Union[np.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]):
        self.wv = self._KeyedVectorsBuilder().from_dict(model).with_weights(weights).build()

    class _KeyedVectorsBuilder:
        def __init__(self):
            pass

        def build(self):
            pass

        def from_dict(self, dictionary):
            pass

        def with_weights(self, weights):
            pass


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
