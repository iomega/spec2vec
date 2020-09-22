import os
import gensim
import numpy
import pytest
from matchms import Spectrum
from spec2vec import SpectrumDocument
from spec2vec.vector_operations import calc_vector
from spec2vec.vector_operations import cosine_similarity
from spec2vec.vector_operations import cosine_similarity_matrix


def test_calc_vector():
    """Test deriving a document vector using a pretrained network."""
    spectrum = Spectrum(mz=numpy.array([100, 150, 200, 250], dtype="float"),
                        intensities=numpy.array([0.1, 0.1, 0.1, 1.0], dtype="float"),
                        metadata={})

    document = SpectrumDocument(spectrum, n_decimals=1)
    model = import_pretrained_model()
    vector = calc_vector(model, document, intensity_weighting_power=0.5, allowed_missing_percentage=1.0)
    expected_vector = numpy.array([0.08982063, -1.43037023, -0.17572929, -0.45750666, 0.44942236,
                                   1.35530729, -1.8305029, -0.36850534, -0.28393048, -0.34192028])
    assert numpy.all(vector == pytest.approx(expected_vector, 1e-5)), "Expected different document vector."


def test_calc_vector_higher_than_allowed_missing_percentage():
    """Test using a pretrained network and a missing word percentage above allowed."""
    spectrum = Spectrum(mz=numpy.array([11.1, 100, 200, 250], dtype="float"),
                        intensities=numpy.array([0.1, 0.1, 0.1, 1.0], dtype="float"),
                        metadata={})

    document = SpectrumDocument(spectrum, n_decimals=1)
    model = import_pretrained_model()
    assert document.words[0] not in model.wv.vocab, "Expected word to be missing from given model."
    with pytest.raises(AssertionError) as msg:
        calc_vector(model, document, intensity_weighting_power=0.5, allowed_missing_percentage=16.0)

    expected_message_part = "Missing percentage is larger than set maximum."
    assert expected_message_part in str(msg.value), "Expected particular error message."


def test_calc_vector_within_allowed_missing_percentage():
    """Test using a pretrained network and a missing word percentage within allowed."""
    spectrum = Spectrum(mz=numpy.array([11.1, 100, 200, 250], dtype="float"),
                        intensities=numpy.array([0.1, 0.1, 0.1, 1.0], dtype="float"),
                        metadata={})

    document = SpectrumDocument(spectrum, n_decimals=1)
    model = import_pretrained_model()
    vector = calc_vector(model, document, intensity_weighting_power=0.5, allowed_missing_percentage=17.0)
    expected_vector = numpy.array([0.12775915, -1.17673617, -0.14598507, -0.40189132, 0.36908966,
                                   1.11608575, -1.46774333, -0.31442554, -0.23168877, -0.29420064])
    assert document.words[0] not in model.wv.vocab, "Expected word to be missing from given model."
    assert numpy.all(vector == pytest.approx(expected_vector, 1e-5)), "Expected different document vector."


def import_pretrained_model():
    """Helper function to import pretrained word2vec model."""
    repository_root = os.path.join(os.path.dirname(__file__), "..")
    model_file = os.path.join(repository_root, "integration-tests", "test_user_workflow_spec2vec.model")
    return gensim.models.Word2Vec.load(model_file)


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_cosine_similarity(numba_compiled):
    """Test cosine similarity score calculation."""
    vector1 = numpy.array([1, 1, 0, 0])
    vector2 = numpy.array([1, 1, 1, 1])

    if numba_compiled:
        score11 = cosine_similarity(vector1, vector1)
        score12 = cosine_similarity(vector1, vector2)
        score22 = cosine_similarity(vector2, vector2)
    else:
        score11 = cosine_similarity.py_func(vector1, vector1)
        score12 = cosine_similarity.py_func(vector1, vector2)
        score22 = cosine_similarity.py_func(vector2, vector2)

    assert score12 == 2 / numpy.sqrt(2 * 4), "Expected different score."
    assert score11 == score22 == 1.0, "Expected different score."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_cosine_similarity_all_zeros(numba_compiled):
    """Test cosine similarity score calculation with empty vector."""
    vector1 = numpy.array([0, 0, 0, 0])
    vector2 = numpy.array([1, 1, 1, 1])

    if numba_compiled:
        score11 = cosine_similarity(vector1, vector1)
        score12 = cosine_similarity(vector1, vector2)
        score22 = cosine_similarity(vector2, vector2)
    else:
        score11 = cosine_similarity.py_func(vector1, vector1)
        score12 = cosine_similarity.py_func(vector1, vector2)
        score22 = cosine_similarity.py_func(vector2, vector2)

    assert score11 == score12 == 0.0, "Expected different score."
    assert score22 == 1.0, "Expected different score."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_cosine_similarity_matrix(numba_compiled):
    """Test cosine similarity scores calculation using int32 input.."""
    vectors1 = numpy.array([[1, 1, 0, 0],
                            [1, 0, 1, 1]], dtype=numpy.int32)
    vectors2 = numpy.array([[0, 1, 1, 0],
                            [0, 0, 1, 1]], dtype=numpy.int32)

    if numba_compiled:
        scores = cosine_similarity_matrix(vectors1, vectors2)
    else:
        scores = cosine_similarity_matrix.py_func(vectors1, vectors2)
    expected_scores = numpy.array([[0.5, 0.],
                                   [0.40824829, 0.81649658]])
    assert scores == pytest.approx(expected_scores, 1e-7), "Expected different scores."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_cosine_similarity_floats_matrix(numba_compiled):
    """Test cosine similarity scores calculation using float64 input.."""
    vectors1 = numpy.array([[1, 1, 0, 0],
                            [1, 0, 1, 1]], dtype=numpy.float64)
    vectors2 = numpy.array([[0, 1, 1, 0],
                            [0, 0, 1, 1]], dtype=numpy.float64)

    if numba_compiled:
        scores = cosine_similarity_matrix(vectors1, vectors2)
    else:
        scores = cosine_similarity_matrix.py_func(vectors1, vectors2)
    expected_scores = numpy.array([[0.5, 0.],
                                   [0.40824829, 0.81649658]])
    assert scores == pytest.approx(expected_scores, 1e-7), "Expected different scores."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_cosine_similarity_matrix_input_cloned(numba_compiled):
    """Test if score implementation clones the input correctly."""
    vectors1 = numpy.array([[2, 2, 0, 0],
                            [2, 0, 2, 2]])
    vectors2 = numpy.array([[0, 2, 2, 0],
                            [0, 0, 2, 2]])

    if numba_compiled:
        cosine_similarity_matrix(vectors1, vectors2)
    else:
        cosine_similarity_matrix.py_func(vectors1, vectors2)

    assert numpy.all(vectors1 == numpy.array([[2, 2, 0, 0],
                                              [2, 0, 2, 2]])), "Expected unchanged input."


def test_differnt_input_vector_lengths():
    """Test if correct error is raised."""
    vector1 = numpy.array([0, 0, 0, 0])
    vector2 = numpy.array([1, 1, 1, 1, 1])

    with pytest.raises(AssertionError) as msg:
        _ = cosine_similarity(vector1, vector2)

    expected_message = "Input vector must have same shape."
    assert expected_message == str(msg.value), "Expected particular error message."
