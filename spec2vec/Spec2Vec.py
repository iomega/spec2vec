from typing import Union
from gensim.models.basemodel import BaseTopicModel
from spec2vec.SpectrumDocument import SpectrumDocument
from spec2vec.utils import cosine_similarity
from .calc_vector import calc_vector


class Spec2Vec:
    """Calculate spec2vec similarity scores between a reference and a query.

    Using a trained model, spectrum documents will be converted into spectrum
    vectors. The spec2vec similarity is then the cosine similarity score between
    two spectrum vectors.

    Example code to calcualte spec2vec similarities between query and reference
    spectrums:

    .. code-block:: python

        import gensim
        from spec2vec import Spec2Vec
        from spec2vec import SpectrumDocument

        # reference_spectrums & query_spectrums loaded from files using https://matchms.readthedocs.io/en/latest/api/matchms.importing.load_from_mgf.html
        references = [SpectrumDocument(s, n_decimals=2) for s in reference_spectrums]
        queries = [SpectrumDocument(s, n_decimals=2) for s in query_spectrums]

        # Import pre-trained word2vec model (alternative: train new model)
        model_file = "path and filename"
        model = gensim.models.Word2Vec.load(model_file)

        # Define similarity_function
        spec2vec = Spec2Vec(model=model, intensity_weighting_power=0.5)

        # Calculate scores on all combinations of references and queries
        scores = list(calculate_scores(references, queries, spec2vec))

        # Filter out self-comparisons
        filtered = [(reference, query, score) for (reference, query, score) in scores if reference != query]

        sorted_by_score = sorted(filtered, key=lambda elem: elem[2], reverse=True)
    """
    def __init__(self, model: BaseTopicModel, intensity_weighting_power: Union[float, int] = 0,
                 allowed_missing_percentage: Union[float, int] = 0):
        """

        Parameters
        ----------
        model:
            Expected input is a gensim word2vec model that has been trained on
            the desired set of spectrum documents.
        intensity_weighting_power:
            Spectrum vectors are a weighted sum of the word vectors. The given
            word intensities will be raised to the given power.
            The default is 0, which means that no weighing will be done.
        allowed_missing_percentage:
            Set the maximum allowed percentage of the document that may be missing
            from the input model. This is measured as percentage of the weighted, missing
            words compared to all word vectors of the document. Default is 0, which
            means no missing words are allowed.
        """
        self.model = model
        self.intensity_weighting_power = intensity_weighting_power
        self.allowed_missing_percentage = allowed_missing_percentage
        self.vector_size = model.wv.vector_size

    def __call__(self, reference: SpectrumDocument, query: SpectrumDocument) -> float:
        """Calculate the spec2vec similaritiy between a reference and a query.

        Parameters
        ----------
        reference:
            Reference spectrum document.
        query:
            Query spectrum document.

        Returns
        -------
        spec2vec_similarity
            Spec2vec similarity score.
        """
        reference_vector = calc_vector(self.model, reference, self.intensity_weighting_power,
                                       self.allowed_missing_percentage)
        query_vector = calc_vector(self.model, query, self.intensity_weighting_power,
                                   self.allowed_missing_percentage)

        return cosine_similarity(reference_vector, query_vector)
