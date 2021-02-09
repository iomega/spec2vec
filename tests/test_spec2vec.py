import os
import gensim
import numpy
import pytest
from matchms import Spectrum
from spec2vec import Spec2Vec
from spec2vec import SpectrumDocument


def test_spec2vec_pair_method():
    """Test if pair of two SpectrumDocuments is handled correctly"""
    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200.]),
                          intensities=numpy.array([0.7, 0.2, 0.1]),
                          metadata={'id': 'spectrum1'})
    spectrum_2 = Spectrum(mz=numpy.array([100, 140, 190.]),
                          intensities=numpy.array([0.4, 0.2, 0.1]),
                          metadata={'id': 'spectrum2'})

    documents = [SpectrumDocument(s, n_decimals=1) for s in [spectrum_1, spectrum_2]]
    model = load_test_model()
    spec2vec = Spec2Vec(model=model, intensity_weighting_power=0.5)
    score01 = spec2vec.pair(documents[0], documents[1])
    assert score01 == pytest.approx(0.9936808, 1e-6)
    score11 = spec2vec.pair(documents[1], documents[1])
    assert score11 == pytest.approx(1.0, 1e-9)


@pytest.mark.parametrize("progress_bar", [True, False])
def test_spec2vec_matrix_method(progress_bar):
    """Test if matrix of 2x2 SpectrumDocuments is handled correctly.
    Run with and without progress bar.
    """
    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200.]),
                          intensities=numpy.array([0.7, 0.2, 0.1]),
                          metadata={'id': 'spectrum1'})
    spectrum_2 = Spectrum(mz=numpy.array([100, 140, 190.]),
                          intensities=numpy.array([0.4, 0.2, 0.1]),
                          metadata={'id': 'spectrum2'})

    documents = [SpectrumDocument(s, n_decimals=1) for s in [spectrum_1, spectrum_2]]
    model = load_test_model()
    spec2vec = Spec2Vec(model=model, intensity_weighting_power=0.5, progress_bar=progress_bar)
    scores = spec2vec.matrix(documents, documents)
    assert scores[0, 0] == pytest.approx(1.0, 1e-9), "Expected different score."
    assert scores[1, 1] == pytest.approx(1.0, 1e-9), "Expected different score."
    assert scores[1, 0] == pytest.approx(0.9936808, 1e-6), "Expected different score."
    assert scores[0, 1] == pytest.approx(0.9936808, 1e-6), "Expected different score."


def test_spec2vec_matrix_method_symmetric():
    """Test if matrix of 2x2 SpectrumDocuments is handled correctly.
    Run with is_symmetric=True.
    """
    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200.]),
                          intensities=numpy.array([0.7, 0.2, 0.1]),
                          metadata={'id': 'spectrum1'})
    spectrum_2 = Spectrum(mz=numpy.array([100, 140, 190.]),
                          intensities=numpy.array([0.4, 0.2, 0.1]),
                          metadata={'id': 'spectrum2'})

    documents = [SpectrumDocument(s, n_decimals=1) for s in [spectrum_1, spectrum_2]]
    model = load_test_model()
    spec2vec = Spec2Vec(model=model, intensity_weighting_power=0.5)
    scores = spec2vec.matrix(documents, documents, is_symmetric=True)
    assert scores[0, 0] == pytest.approx(1.0, 1e-9), "Expected different score."
    assert scores[1, 1] == pytest.approx(1.0, 1e-9), "Expected different score."
    assert scores[1, 0] == pytest.approx(0.9936808, 1e-6), "Expected different score."
    assert scores[0, 1] == pytest.approx(0.9936808, 1e-6), "Expected different score."


def test_spec2vec_matrix_method_symmetric_wrong_entry():
    """Test if matrix of 2x2 SpectrumDocuments is handled correctly.
    Run with is_symmetric=True but non symmetric entries.
    """
    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200.]),
                          intensities=numpy.array([0.7, 0.2, 0.1]),
                          metadata={'id': 'spectrum1'})
    spectrum_2 = Spectrum(mz=numpy.array([100, 140, 190.]),
                          intensities=numpy.array([0.4, 0.2, 0.1]),
                          metadata={'id': 'spectrum2'})

    documents1 = [SpectrumDocument(s, n_decimals=1) for s in [spectrum_1, spectrum_2]]
    documents2 = [SpectrumDocument(s, n_decimals=1) for s in [spectrum_2, spectrum_1]]
    model = load_test_model()
    spec2vec = Spec2Vec(model=model, intensity_weighting_power=0.5)
    expected_msg = "Expected references to be equal to queries for is_symmetric=True"
    with pytest.raises(AssertionError) as msg:
        _ = spec2vec.matrix(documents1, documents2, is_symmetric=True)
    assert expected_msg in str(msg), "Expected different exception message"


def load_test_model():
    """Load pretrained Word2Vec model."""
    repository_root = os.path.join(os.path.dirname(__file__), "..")
    model_file = os.path.join(repository_root, "integration-tests", "test_user_workflow_spec2vec.model")
    assert os.path.isfile(model_file), "Expected file not found."
    return gensim.models.Word2Vec.load(model_file)
