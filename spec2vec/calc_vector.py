from typing import Union
import numpy
from gensim.models.basemodel import BaseTopicModel
from spec2vec.Document import Document


def calc_vector(model: BaseTopicModel, document: Document,
                intensity_weighting_power: Union[float, int] = 0) -> numpy.ndarray:
    """Compute document vector form individual word vectors (and weights).

    Parameters
    ----------
    model
        Pretrained word2vec model to convert words into vectors.
    document
        Document containing document.words and document.weights.
    intensity_weighting_power
        Specify to what power weights should be raised. The default is 0, which
        means that no weighing will be done.

    Returns
    -------
    vector
        Vector representing the input document in latent space.
    """
    word_vectors = model.wv[document.words]
    weights = numpy.asarray(document.weights).reshape(len(document), 1)
    weights_raised = numpy.power(weights, intensity_weighting_power)
    weights_raised_tiled = numpy.tile(weights_raised, (1, model.wv.vector_size))
    vector = numpy.sum(word_vectors * weights_raised_tiled, 0)
    return vector
