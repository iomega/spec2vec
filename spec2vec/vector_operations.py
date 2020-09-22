from typing import Union
import numba
import numpy
from gensim.models.basemodel import BaseTopicModel
from spec2vec.Document import Document


def calc_vector(model: BaseTopicModel, document: Document,
                intensity_weighting_power: Union[float, int] = 0,
                allowed_missing_percentage: Union[float, int] = 0) -> numpy.ndarray:
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
        words compared to all word vectors of the document. Default is 0, which
        means no missing words are allowed.

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
            weights_missing = numpy.array([document.weights[i] for i in idx_not_in_model])
            weights_missing_raised = numpy.power(weights_missing, intensity_weighting_power)
            missing_percentage = 100 * weights_missing_raised.sum() / (weights_raised.sum()
                                                                       + weights_missing_raised.sum())
            print("Found {} word(s) missing in the model.".format(len(idx_not_in_model)),
                  "Weighted missing percentage not covered by the given model is {:.2f}%.".format(missing_percentage))

            message = ("Missing percentage is larger than set maximum.",
                       "Consider retraining the used model or increasing the allowed percentage.")
            assert missing_percentage <= allowed_missing_percentage, message

    idx_not_in_model = [i for i, x in enumerate(document.words) if x not in model.wv.vocab]
    words_in_model = [x for i, x in enumerate(document.words) if i not in idx_not_in_model]
    weights_in_model = numpy.asarray([x for i, x in enumerate(document.weights)
                                      if i not in idx_not_in_model]).reshape(len(words_in_model), 1)

    word_vectors = model.wv[words_in_model]
    weights_raised = numpy.power(weights_in_model, intensity_weighting_power)

    _check_model_coverage()

    weights_raised_tiled = numpy.tile(weights_raised, (1, model.wv.vector_size))
    vector = numpy.sum(word_vectors * weights_raised_tiled, 0)
    return vector


@numba.njit
def cosine_similarity_matrix(vectors_1: numpy.ndarray, vectors_2: numpy.ndarray) -> numpy.ndarray:
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
    vectors_1 = vectors_1.astype(numpy.float64)  # Numba dot only accepts float or complex arrays
    vectors_2 = vectors_2.astype(numpy.float64)
    norm_1 = numpy.sqrt(numpy.sum(vectors_1**2, axis=1))
    norm_2 = numpy.sqrt(numpy.sum(vectors_2**2, axis=1))
    for i in range(vectors_1.shape[0]):
        vectors_1[i] = vectors_1[i] / norm_1[i]
    for i in range(vectors_2.shape[0]):
        vectors_2[i] = vectors_2[i] / norm_2[i]
    return numpy.dot(vectors_1, vectors_2.T)


@numba.njit
def cosine_similarity(vector1: numpy.ndarray, vector2: numpy.ndarray) -> numpy.float64:
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
        cosine_score = prod12 / numpy.sqrt(prod11 * prod22)
    return numpy.float64(cosine_score)
