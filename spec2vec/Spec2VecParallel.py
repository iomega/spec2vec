import numpy
import scipy
from .calc_vector import calc_vector


class Spec2VecParallel:
    """Calculate spec2vec similarity scores between all references and queries.

    Using a trained model, spectrum documents will be converted into spectrum
    vectors. The spec2vec similarity is then the cosine similarity score between
    two spectrum vectors.
    """
    def __init__(self, model, intensity_weighting_power=0):
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
        """
        self.model = model
        self.intensity_weighting_power = intensity_weighting_power
        self.vector_size = model.wv.vector_size

    def __call__(self, references, queries) -> numpy.array:
        """Calculate the spec2vec similarities between all references and queries.

        Parameters
        ----------
        references : list of SpectrumDocuments
            Reference spectrum documents.
        queries : list of SpectrumDocuments
            Query spectrum documents.

        Returns
        -------
        spec2vec_similarity
            Array of spec2vec similarity scores.
        """
        n_rows = len(references)
        reference_vectors = numpy.empty((n_rows, self.vector_size), dtype="float")
        for index_reference, reference in enumerate(references):
            reference_vectors[index_reference, 0:self.vector_size] = calc_vector(self.model,
                                                                                 reference,
                                                                                 self.intensity_weighting_power)
        n_cols = len(queries)
        query_vectors = numpy.empty((n_cols, self.vector_size), dtype="float")
        for index_query, query in enumerate(queries):
            query_vectors[index_query, 0:self.vector_size] = calc_vector(self.model,
                                                                         query,
                                                                         self.intensity_weighting_power)

        cdist = scipy.spatial.distance.cdist(reference_vectors, query_vectors, "cosine")

        return 1 - cdist
