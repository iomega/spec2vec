import re
from typing import List, Union
import numpy as np
from gensim.models import Word2Vec
from matchms import Spectrum
from matchms.similarity.BaseSimilarity import BaseSimilarity
from tqdm import tqdm
from spec2vec.SpectrumDocument import SpectrumDocument
from spec2vec.vector_operations import (calc_vector, cosine_similarity,
                                        cosine_similarity_matrix)


class Spec2Vec(BaseSimilarity):
    """Calculate spec2vec similarity scores between a reference and a query.

    Using a trained model, spectrum documents will be converted into spectrum
    vectors. The spec2vec similarity is then the cosine similarity score between
    two spectrum vectors.

    The following code example shows how to calculate spec2vec similarities
    between query and reference spectrums. It uses a dummy model that can be found at
    :download:`../integration-tests/test_user_workflow_spec2vec.model </../integration-tests/test_user_workflow_spec2vec.model>`
    and a small test dataset that can be found at
    :download:`../tests/pesticides.mgf </../tests/pesticides.mgf>`.

    .. testcode::

        import os
        import gensim
        from matchms import calculate_scores
        from matchms.filtering import add_losses
        from matchms.filtering import default_filters
        from matchms.filtering import normalize_intensities
        from matchms.filtering import require_minimum_number_of_peaks
        from matchms.filtering import select_by_intensity
        from matchms.filtering import select_by_mz
        from matchms.importing import load_from_mgf
        from spec2vec import Spec2Vec

        def spectrum_processing(s):
            '''This is how a user would typically design his own pre- and post-
            processing pipeline.'''
            s = default_filters(s)
            s = normalize_intensities(s)
            s = select_by_mz(s, mz_from=0, mz_to=1000)
            s = select_by_intensity(s, intensity_from=0.01)
            s = add_losses(s, loss_mz_from=10.0, loss_mz_to=200.0)
            s = require_minimum_number_of_peaks(s, n_required=5)
            return s

        spectrums_file = os.path.join(os.getcwd(), "..", "tests", "pesticides.mgf")

        # Load data and apply the above defined filters to the data
        spectrums = [spectrum_processing(s) for s in load_from_mgf(spectrums_file)]

        # Omit spectrums that didn't qualify for analysis
        spectrums = [s for s in spectrums if s is not None]

        # Load pretrained model (here dummy model)
        model_file = os.path.join(os.getcwd(), "..", "integration-tests", "test_user_workflow_spec2vec.model")
        model = gensim.models.Word2Vec.load(model_file)

        # Define similarity_function
        spec2vec = Spec2Vec(model=model, intensity_weighting_power=0.5)

        # Calculate scores on all combinations of references and queries
        scores = calculate_scores(spectrums[10:], spectrums[:10], spec2vec)

        # Select top-10 candidates for first query spectrum
        spectrum0_top10 = scores.scores_by_query(spectrums[0], sort=True)[:10]

        # Display spectrum IDs for top-10 matches
        print([s[0].metadata['spectrumid'] for s in spectrum0_top10])

    Should output

    .. testoutput::

        ['CCMSLIB00001058300', 'CCMSLIB00001058289', 'CCMSLIB00001058303', ...

    """
    def __init__(self, model: Word2Vec, intensity_weighting_power: Union[float, int] = 0,
                 allowed_missing_percentage: Union[float, int] = 10, progress_bar: bool = False):
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
            words compared to all word vectors of the document. Default is 10, which
            means up to 10% missing words are allowed. If more words are missing from
            the model, an empty embedding will be returned (leading to similarities of 0)
            and a warning is raised.
        progress_bar:
            Set to True to monitor the embedding creating with a progress bar.
            Default is False.
        """
        self.model = model
        self.n_decimals = self._get_word_decimals(self.model)
        self.intensity_weighting_power = intensity_weighting_power
        self.allowed_missing_percentage = allowed_missing_percentage
        self.vector_size = model.wv.vector_size
        self.disable_progress_bar = not progress_bar

    def pair(self, reference: Union[SpectrumDocument, Spectrum],
             query: Union[SpectrumDocument, Spectrum]) -> float:
        """Calculate the spec2vec similaritiy between a reference and a query.

        Parameters
        ----------
        reference:
            Reference spectrum or spectrum document.
        query:
            Query spectrum or spectrum document.

        Returns
        -------
        spec2vec_similarity
            Spec2vec similarity score.
        """
        reference_vector = self._calculate_embedding(reference)
        query_vector = self._calculate_embedding(query)

        return cosine_similarity(reference_vector, query_vector)

    def matrix(self, references: Union[List[SpectrumDocument], List[Spectrum]],
               queries: Union[List[SpectrumDocument], List[Spectrum]],
               is_symmetric: bool = False) -> np.ndarray:
        """Calculate the spec2vec similarities between all references and queries.

        Parameters
        ----------
        references:
            Reference spectrums or spectrum documents.
        queries:
            Query spectrums or spectrum documents.
        is_symmetric:
            Set to True if references == queries to speed up calculation about 2x.
            Uses the fact that in this case score[i, j] = score[j, i]. Default is False.

        Returns
        -------
        spec2vec_similarity
            Array of spec2vec similarity scores.
        """
        n_rows = len(references)
        reference_vectors = np.empty((n_rows, self.vector_size), dtype="float")
        for index_reference, reference in enumerate(tqdm(references, desc='Calculating vectors of reference spectrums',
                                                         disable=self.disable_progress_bar)):
            reference_vectors[index_reference, 0:self.vector_size] = self._calculate_embedding(reference)

        n_cols = len(queries)
        if is_symmetric:
            assert np.all(references == queries), \
                "Expected references to be equal to queries for is_symmetric=True"
            query_vectors = reference_vectors
        else:
            query_vectors = np.empty((n_cols, self.vector_size), dtype="float")
            for index_query, query in enumerate(tqdm(queries, desc='Calculating vectors of query spectrums',
                                                     disable=self.disable_progress_bar)):
                query_vectors[index_query, 0:self.vector_size] = self._calculate_embedding(query)

        spec2vec_similarity = cosine_similarity_matrix(reference_vectors, query_vectors)

        return spec2vec_similarity

    @staticmethod
    def _get_word_decimals(model):
        """Read the decimal rounding that was used to train the model"""
        word_regex = r"[a-z]{4}@[0-9]{1,5}."
        example_word = next(iter(model.wv.key_to_index))

        return len(re.split(word_regex, example_word)[-1])

    def _calculate_embedding(self, spectrum_in: Union[SpectrumDocument, Spectrum]):
        """Generate Spec2Vec embedding vectors from input spectrum (or SpectrumDocument)"""
        if isinstance(spectrum_in, Spectrum):
            spectrum_in = SpectrumDocument(spectrum_in, n_decimals=self.n_decimals)
        elif isinstance(spectrum_in, SpectrumDocument):
            assert spectrum_in.n_decimals == self.n_decimals, \
                "Decimal rounding of input data does not agree with model vocabulary."
        else:
            raise ValueError("Expected input type to be Spectrum or SpectrumDocument")
        return calc_vector(self.model,
                           spectrum_in,
                           self.intensity_weighting_power,
                           self.allowed_missing_percentage)
