from typing import Union
import scipy
from .calc_vector import calc_vector


class Spec2Vec:
    """Calculate spec2vec similarity scores between a reference and a query.

    Using a trained model, spectrum documents will be converted into spectrum
    vectors. The spec2vec similarity is then the cosine similarity score between
    two spectrum vectors.
    """
    def __init__(self, model, intensity_weighting_power=0,
                 allowed_missing_fraction: Union[float, int] = 0):
        """

        Parameters
        ----------
        model : gensim word2vec model
            Expecgted input is a gensim word2vec model that has been trained on
            the desired set of spectrum documents.
        intensity_weighting_power : float, optional
            Spectrum vectors are a weighted sum of the word vectors. The given
            word intensities will be raised to the given power.
            The default is 0, which means that no weighing will be done.
        allowed_missing_fraction:
            Set the maximum allowed fraction (in percent) of the document that may
            be missing from the input model. This is measured as fraction of the
            missing words compared to all word vectors of the document. Default is
            0, which means no missing words are allowed.
        """
        self.model = model
        self.intensity_weighting_power = intensity_weighting_power
        self.allowed_missing_fraction = allowed_missing_fraction
        self.vector_size = model.wv.vector_size

    def __call__(self, reference, query) -> float:
        """Calculate the spec2vec similaritiy between a reference and a query.

        Parameters
        ----------
        reference : SpectrumDocuments
            Reference spectrum documen.
        query : SpectrumDocuments
            Query spectrum document.

        Returns
        -------
        spec2vec_similarity
            Spec2vec similarity score.
        """
        reference_vector = calc_vector(self.model, reference,self.intensity_weighting_power,
                                       self.allowed_missing_fraction)
        query_vector = calc_vector(self.model, query, self.intensity_weighting_power,
                                   self.allowed_missing_fraction)
        cdist = scipy.spatial.distance.cosine(reference_vector, query_vector)

        return 1 - cdist
