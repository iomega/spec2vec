import numpy


def calc_vector(model, document, intensity_weighting_power=1, allowed_missing_fraction=0) -> numpy.array:
    """Compute document vector form individual word vectors (and weights).

    Parameters
    ----------
    model : gensim word2vec model
        Pretrained word2vec model to convert words into vectors.
    document : DocumentType
        Document containing document.words and document.weights.
    intensity_weighting_power : float
        Specify to what power weights should be raised. The default is 0, which
        means that no weighing will be done.
    allowed_missing_fraction : float within [0,100]
        Set the maximum allowed fraction (in percent) of the document that may be missing
        from the input model. This is measured as fraction of the missing words compared
        to all word vectors of the document. Default is 0, which means no missing
        words are allowed.

    Returns
    -------
    vector:
        Vector representing the input document in latent space. Will return None
        if the missing fraction of the document in the model is > allowed_missing_fraction.
    """
    assert max(document.weights) <= 1.0, "Weights are not normalized to unity as expected."

    def _sufficient_model_coverage():
        """Return True if model covers enough of the document words."""
        if len(idx_not_in_model) > 0:
            weights_missing = numpy.array([document.weights[i] for i in idx_not_in_model])
            weights_missing_raised = numpy.power(weights_missing, intensity_weighting_power)
            missing_fraction = 100 * weights_missing_raised.sum() / (weights_raised.sum()
                                                                     + weights_missing_raised.sum())
            print("Found {} word(s) missing in the model.".format(len(idx_not_in_model)),
                  "Weighted fraction not covered is {:.2f}%.".format(missing_fraction))
            if missing_fraction > allowed_missing_fraction:
                print("Missing fraction is larger than set maximum.",
                      "Consider retraining the used model.")
                return False
        return True

    idx_not_in_model = [i for i, x in enumerate(document.words) if x not in model.wv.vocab]
    words_in_model = [x for i, x in enumerate(document.words) if i not in idx_not_in_model]
    weights_in_model = numpy.asarray([x for i, x in enumerate(document.weights)
                                      if i not in idx_not_in_model]).reshape(len(words_in_model), 1)

    word_vectors = model.wv[words_in_model]
    weights_raised = numpy.power(weights_in_model, intensity_weighting_power)

    if _sufficient_model_coverage():
        weights_raised_tiled = numpy.tile(weights_raised, (1, model.wv.vector_size))
        vector = numpy.sum(word_vectors * weights_raised_tiled, 0)
        return vector

    return None
