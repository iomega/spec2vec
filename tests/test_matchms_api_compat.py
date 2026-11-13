"""Spec2Vec compatibility smoke tests for both matchms 0.x and 1.0.

Run these tests in both matchms environments in CI. No model download or
Gensim training is necessary: the tiny word vectors are deterministic.
"""
import numpy as np
import pytest
from matchms import Spectrum, calculate_scores
from spec2vec import Spec2Vec, SpectrumDocument
from spec2vec._matchms_compat import MATCHMS_V1, Scores
from spec2vec.vector_operations import cosine_similarity_matrix


class FakeKeyedVectors:
    vector_size = 2
    key_to_index = {"peak@100.0": 0, "peak@200.0": 1}
    vectors = {
        "peak@100.0": np.array([1.0, 0.0]),
        "peak@200.0": np.array([0.0, 1.0]),
    }

    def __getitem__(self, keys):
        return np.stack([self.vectors[key] for key in keys])


class FakeModel:
    wv = FakeKeyedVectors()


def _spectrum(mz, intensities):
    return Spectrum(
        mz=np.asarray(mz, dtype=float),
        intensities=np.asarray(intensities, dtype=float),
        metadata={"id": "compatibility-test"},
    )


def test_spectrum_document_and_pair():
    spectrum = _spectrum([100.0, 200.0], [1.0, 0.5])
    document = SpectrumDocument(spectrum, n_decimals=1)
    assert document.words == ["peak@100.0", "peak@200.0"]

    similarity = Spec2Vec(FakeModel())
    assert similarity.pair(spectrum, spectrum) == pytest.approx(1.0)
    assert similarity.pair(document, document) == pytest.approx(1.0)


def test_matrix_contract_and_calculate_scores():
    references = [_spectrum([100.0], [1.0]), _spectrum([200.0], [1.0])]
    queries = [_spectrum([100.0], [1.0])]
    similarity = Spec2Vec(FakeModel())

    matrix = similarity.matrix(references, queries, progress_bar=False)
    if MATCHMS_V1:
        assert isinstance(matrix, Scores)
        numeric = matrix.to_array("score")
    else:
        assert isinstance(matrix, np.ndarray)
        numeric = matrix
    np.testing.assert_allclose(numeric, [[1.0], [0.0]])

    # New positional score_fields and old positional array_type both work.
    modern_call = similarity.matrix(references, queries, ("score",), False)
    old_call = similarity.matrix(references, queries, "numpy", False)
    for result in (modern_call, old_call):
        values = result.to_array("score") if MATCHMS_V1 else result
        np.testing.assert_allclose(values, numeric)

    actual = calculate_scores(references, queries, similarity)
    if MATCHMS_V1:
        assert isinstance(actual, Scores)
        np.testing.assert_allclose(actual.to_array("score"), numeric)
    else:
        # Old calculate_scores wraps the raw matrix in its own Scores class.
        assert actual is not None


def test_symmetric_and_score_fields():
    spectra = [_spectrum([100.0], [1.0]), _spectrum([200.0], [1.0])]
    similarity = Spec2Vec(FakeModel())
    actual = similarity.matrix(spectra, progress_bar=False, score_fields=("score",))
    numeric = actual.to_array("score") if MATCHMS_V1 else actual
    np.testing.assert_allclose(numeric, np.eye(2))

    with pytest.raises(ValueError, match="score_fields"):
        similarity.matrix(spectra, score_fields=("matches",))


def test_zero_embedding_gives_zero_similarity():
    vectors = np.array([[0.0, 0.0], [1.0, 0.0]])
    numeric = cosine_similarity_matrix(vectors, vectors)
    np.testing.assert_allclose(numeric, [[0.0, 0.0], [0.0, 1.0]])
    assert np.isfinite(numeric).all()

    known = _spectrum([100.0], [1.0])
    missing = _spectrum([300.0], [1.0])
    similarity = Spec2Vec(FakeModel())
    assert similarity.pair(known, missing) == pytest.approx(0.0)
    result = similarity.matrix([known, missing], [known], progress_bar=False)
    score_values = result.to_array("score") if MATCHMS_V1 else result
    np.testing.assert_allclose(score_values, [[1.0], [0.0]])


def test_sparse_matrix_output():
    spectra = [_spectrum([100.0], [1.0]), _spectrum([200.0], [1.0])]
    similarity = Spec2Vec(FakeModel())
    result = similarity.matrix(spectra, spectra, array_type="sparse", progress_bar=False)
    if MATCHMS_V1:
        assert isinstance(result, Scores)
        assert result.is_sparse
        np.testing.assert_allclose(result.to_array("score"), np.eye(2))
    else:
        from sparsestack import \
            StackedSparseArray  # pylint: disable=import-outside-toplevel
        assert isinstance(result, StackedSparseArray)
