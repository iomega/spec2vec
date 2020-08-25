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

    Parameters
    ----------
    vectors_1
        Numpy array of vectors. vectors_1.shape[0] is number of vectors, vectors_1.shape[1]
        is vector dimension.
    vectors_2
        Numpy array of vectors. vectors_2.shape[0] is number of vectors, vectors_2.shape[1]
        is vector dimension.
    """
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
def cosine_similarity(u: numpy.ndarray, v: numpy.ndarray) -> numpy.float64:
    """Calculate cosine similarity score.

    Parameters
    ----------
    u
        Input vector.
    v
        Input vector.

    Returns
    -------
    cosine_similarity
        The Cosine similarity score between vectors `u` and `v`.
    """
    assert u.shape[0] == v.shape[0], "Input vector must have same shape."
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]
    cosine_score = 0
    if uu != 0 and vv != 0:
        cosine_score = uv / numpy.sqrt(uu * vv)
    return numpy.float64(cosine_score)
