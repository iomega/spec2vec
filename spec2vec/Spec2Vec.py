import re
from typing import List, Union
import numpy as np
from gensim.models import Word2Vec
from matchms import Spectrum
from tqdm import tqdm
from spec2vec._matchms_compat import MATCHMS_V1, BaseSimilarity, Scores
from spec2vec.serialization import Word2VecLight
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
            s = require_minimum_number_of_peaks(s, n_required=5)
            return s

        spectrums_file = os.path.join(os.getcwd(), "..", "tests", "data", "pesticides.mgf")

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

        # The two generations of matchms expose different Scores APIs.
        if hasattr(scores, "score_fields"):  # matchms >= 1.0
            import numpy as np
            top_indices = np.argsort(scores.to_array("score")[:, 0])[::-1][:10]
            top_matches = [spectrums[10:][i] for i in top_indices]
        else:  # matchms 0.x
            top_matches = [entry[0] for entry in
                           scores.scores_by_query(spectrums[0], sort=True)[:10]]

        top10_ids = [s.metadata["spectrum_id"] for s in top_matches]

    """

    is_commutative = True
    score_datatype = np.float64
    score_fields = ("score",)

    def __init__(self, model: Union[Word2Vec, Word2VecLight], intensity_weighting_power: Union[float, int] = 0,
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

    # pylint: disable=arguments-differ,too-many-arguments,too-many-locals
    def matrix(self, references: Union[List[SpectrumDocument], List[Spectrum]],
               queries: Union[List[SpectrumDocument], List[Spectrum], None] = None,
               score_fields=None,
               progress_bar: Union[bool, None] = None,
               *, array_type: str = "numpy", is_symmetric: bool = False):
        """Calculate Spec2Vec similarities between reference and query documents.

        The matrix uses reference rows and query columns. With matchms >= 1.0,
        a :class:`matchms.scores.Scores` container is returned. With matchms
        0.x, the legacy NumPy or StackedSparseArray result is retained.

        Parameters
        ----------
        references
            Reference spectra or spectrum documents.
        queries
            Query spectra or documents. ``None`` means compare references with
            themselves (and reuse their embeddings).
        array_type
            ``"numpy"`` (default) or ``"sparse"``. Sparse legacy output is a
            StackedSparseArray; under matchms 1.0 it is a sparse Scores object.
        is_symmetric
            Legacy option for identical reference and query collections.
        score_fields
            matchms 1.0 option. Only ``("score",)`` is supported.
        progress_bar
            Override the progress bar setting from the constructor for this call.
        """
        # Recognize old positional usage matrix(refs, queries, "numpy", True).
        # Keyword array_type/is_symmetric continues to work on both matchms APIs.
        if isinstance(score_fields, str) and score_fields in ("numpy", "sparse"):
            array_type = score_fields
            score_fields = None
            if progress_bar is not None:
                is_symmetric = progress_bar
                progress_bar = None
        if score_fields is not None and tuple(score_fields) != ("score",):
            raise ValueError("Spec2Vec supports only score_fields=('score',).")
        if array_type not in ("numpy", "sparse"):
            raise ValueError("array_type must be 'numpy' or 'sparse'.")

        if queries is None:
            queries = references
            is_symmetric = True
        if is_symmetric:
            if references is not queries and not np.array_equal(references, queries):
                # Preserve the exception type and message used by matchms 0.x callers.
                raise AssertionError(
                    "Expected references to be equal to queries for is_symmetric=True"
                )
        show_progress = not self.disable_progress_bar if progress_bar is None else progress_bar

        n_rows = len(references)
        reference_vectors = np.empty((n_rows, self.vector_size), dtype=float)
        for i, reference in enumerate(tqdm(
                references, desc="Calculating reference vectors", disable=not show_progress)):
            reference_vectors[i, :] = self._calculate_embedding(reference)

        if is_symmetric:
            query_vectors = reference_vectors
        else:
            query_vectors = np.empty((len(queries), self.vector_size), dtype=float)
            for j, query in enumerate(tqdm(
                    queries, desc="Calculating query vectors", disable=not show_progress)):
                query_vectors[j, :] = self._calculate_embedding(query)

        similarities = cosine_similarity_matrix(reference_vectors, query_vectors)

        if MATCHMS_V1:
            if array_type == "sparse":
                from scipy.sparse import \
                    coo_array  # pylint: disable=import-outside-toplevel
                return Scores({"score": coo_array(similarities)})
            return Scores({"score": similarities})

        if array_type == "sparse":
            # Only needed for the old matchms scoring API.
            from sparsestack import \
                StackedSparseArray  # pylint: disable=import-outside-toplevel
            sparse = StackedSparseArray(len(references), len(queries))
            sparse.add_dense_matrix(similarities, "")
            return sparse
        return similarities

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
