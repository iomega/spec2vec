import logging
from typing import Union
import numba
import numpy as np
from gensim.models.basemodel import BaseTopicModel
from spec2vec.Document import Document


logger = logging.getLogger("spec2vec")


def calc_vector(model: BaseTopicModel, document: Document,
                intensity_weighting_power: Union[float, int] = 0,
                allowed_missing_percentage: Union[float, int] = 10) -> np.ndarray:
    """Compute document vector as a (weighted) sum of individual word vectors.

    Parameters
    ----------
    model
        Pretrained word2vec model to convert words into vectors.
    document
        Document containing document.words and document.weights.
    intensity_weighting_power
        Specify to what power weights should be raised. The default is 0, which
        means that no weighing will be done.
    allowed_missing_percentage:
        Set the maximum allowed percentage of the document that may be missing
        from the input model. This is measured as percentage of the weighted, missing
        words compared to all word vectors of the document. Default is 10, which
        means up to 10% missing words are allowed. If more words are missing from
        the model, an empty embedding will be returned (leading to similarities of 0)
        and a warning is raised.

    Returns
    -------
    vector
        Vector representing the input document in latent space. Will return None
        if the missing percentage of the document in the model is > allowed_missing_percentage.
    """
    assert max(document.weights) <= 1.0, "Weights are not normalized to unity as expected."
    assert 0 <= allowed_missing_percentage <= 100.0, "allowed_missing_percentage must be within [0,100]"

    def _check_model_coverage():
        """Return True if model covers enough of the document words."""
        if len(idx_not_in_model) > 0:
            weights_missing = np.array([document.weights[i] for i in idx_not_in_model])
            weights_missing_raised = np.power(weights_missing, intensity_weighting_power)
            missing_percentage = 100 * weights_missing_raised.sum() / (weights_raised.sum()
                                                                       + weights_missing_raised.sum())
            msg = (f"Found {len(idx_not_in_model)} word(s) missing in the model.",
                   f"Weighted missing percentage not covered by the given model is {missing_percentage:.2f}%.")
            logger.info(msg)

            if missing_percentage > allowed_missing_percentage:
                msg = (f"Missing percentage ({missing_percentage:.2f}%) is above set maximum. An empty vector will be returned.",
                       "Consider retraining the used model or change the `allowed_missing_percentage`.")
                logger.warning(msg)
                return False
        return True

    idx_not_in_model = [i for i, x in enumerate(document.words) if x not in model.wv.key_to_index]
    if len(idx_not_in_model) == len(document.words):
        msg = ("Spectrum without peaks known by the used model. An empty vector will be returned.",
               "Consider retraining the used model or make sure decimal rounding is correct.")
        logger.warning(msg)
        return np.zeros(model.wv.vector_size)

    words_in_model = [x for i, x in enumerate(document.words) if i not in idx_not_in_model]
    weights_in_model = np.asarray([x for i, x in enumerate(document.weights)
                                   if i not in idx_not_in_model]).reshape(len(words_in_model), 1)

    word_vectors = model.wv[words_in_model]
    weights_raised = np.power(weights_in_model, intensity_weighting_power)

    if _check_model_coverage() is True:
        weights_raised_tiled = np.tile(weights_raised, (1, model.wv.vector_size))
        return np.sum(word_vectors * weights_raised_tiled, 0)
    return np.zeros(model.wv.vector_size)


@numba.njit
def cosine_similarity_matrix(vectors_1: np.ndarray, vectors_2: np.ndarray) -> np.ndarray:
    """Fast implementation of cosine similarity between two arrays of vectors.

    For example:

    .. code-block:: python

        import numpy as np
        from spec2vec.vector_operations import cosine_similarity_matrix

        vectors_1 = np.array([[1, 1, 0, 0],
                              [1, 0, 1, 1]])
        vectors_2 = np.array([[0, 1, 1, 0],
                              [0, 0, 1, 1]])
        similarity_matrix = cosine_similarity_matrix(vectors_1, vectors_2)


    Parameters
    ----------
    vectors_1
        Numpy array of vectors. vectors_1.shape[0] is number of vectors, vectors_1.shape[1]
        is vector dimension.
    vectors_2
        Numpy array of vectors. vectors_2.shape[0] is number of vectors, vectors_2.shape[1]
        is vector dimension.
    """
    assert vectors_1.shape[1] == vectors_2.shape[1], "Input vectors must have same shape."
    vectors_1 = vectors_1.astype(np.float64)  # Numba dot only accepts float or complex arrays
    vectors_2 = vectors_2.astype(np.float64)
    norm_1 = np.sqrt(np.sum(vectors_1**2, axis=1))
    norm_2 = np.sqrt(np.sum(vectors_2**2, axis=1))
    for i in range(vectors_1.shape[0]):
        vectors_1[i] = vectors_1[i] / norm_1[i]
    for i in range(vectors_2.shape[0]):
        vectors_2[i] = vectors_2[i] / norm_2[i]
    return np.dot(vectors_1, vectors_2.T)


@numba.njit
def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> np.float64:
    """Calculate cosine similarity between two input vectors.

    For example:

    .. testcode::

        import numpy as np
        from spec2vec.vector_operations import cosine_similarity

        vector1 = np.array([1, 1, 0, 0])
        vector2 = np.array([1, 1, 1, 1])
        print("Cosine similarity: {:.3f}".format(cosine_similarity(vector1, vector2)))

    Should output

    .. testoutput::

        Cosine similarity: 0.707

    Parameters
    ----------
    vector1
        Input vector. Can be array of integers or floats.
    vector2
        Input vector. Can be array of integers or floats.
    """
    assert vector1.shape[0] == vector2.shape[0], "Input vector must have same shape."
    prod12 = 0
    prod11 = 0
    prod22 = 0
    for i in range(vector1.shape[0]):
        prod12 += vector1[i] * vector2[i]
        prod11 += vector1[i] * vector1[i]
        prod22 += vector2[i] * vector2[i]
    cosine_score = 0
    if prod11 != 0 and prod22 != 0:
        cosine_score = prod12 / np.sqrt(prod11 * prod22)
    return np.float64(cosine_score)
