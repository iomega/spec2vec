import os
import gensim
import numpy
import pytest
from matchms import Spectrum
from spec2vec import SpectrumDocument
from spec2vec.calc_vector import calc_vector


def test_calc_vector():
    """Test deriving a document vector using a pretrained network."""
    spectrum = Spectrum(mz=numpy.array([100, 150, 200, 250], dtype="float"),
                        intensities=numpy.array([0.1, 0.1, 0.1, 1.0], dtype="float"),
                        metadata={})

    document = SpectrumDocument(spectrum, n_decimals=1)
    model = import_pretrained_model()
    vector = calc_vector(model, document, intensity_weighting_power=0.5, allowed_missing_percentage=1.0)
    expected_vector = numpy.array([-0.0746446, -0.12505581, 0.14092048, 0.06775726, 0.01891184,
                                   -0.04799871, -0.01244692, -0.07346986, -0.00246345, -0.07062957])
    assert numpy.all(vector == pytest.approx(expected_vector, 1e-5)), "Expected different document vector."


def test_calc_vector_higher_than_allowed_missing_percentage():
    """Test using a pretrained network and a missing word percentage above allowed."""
    spectrum = Spectrum(mz=numpy.array([100, 111.1, 200, 250], dtype="float"),
                        intensities=numpy.array([0.1, 0.1, 0.1, 1.0], dtype="float"),
                        metadata={})

    document = SpectrumDocument(spectrum, n_decimals=1)
    model = import_pretrained_model()
    assert document.words[1] not in model.wv.vocab, "Expected word to be missing from given model."
    with pytest.raises(AssertionError) as msg:
        calc_vector(model, document, intensity_weighting_power=0.5, allowed_missing_percentage=1.0)

    expected_message_part = "Missing percentage is larger than set maximum."
    assert expected_message_part in str(msg.value), "Expected particular error message."


def test_calc_vector_within_allowed_missing_percentage():
    """Test using a pretrained network and a missing word percentage within allowed."""
    spectrum = Spectrum(mz=numpy.array([100, 111.1, 200, 250], dtype="float"),
                        intensities=numpy.array([0.1, 0.1, 0.1, 1.0], dtype="float"),
                        metadata={})

    document = SpectrumDocument(spectrum, n_decimals=1)
    model = import_pretrained_model()
    vector = calc_vector(model, document, intensity_weighting_power=0.5, allowed_missing_percentage=20.0)
    expected_vector = numpy.array([-0.0760859, -0.09670882, 0.10740811, 0.04696679, 0.00893024,
                                   -0.04496526, -0.0079267, -0.05153507, -0.00433566, -0.06211582])
    assert document.words[1] not in model.wv.vocab, "Expected word to be missing from given model."
    assert numpy.all(vector == pytest.approx(expected_vector, 1e-5)), "Expected different document vector."


def import_pretrained_model():
    repository_root = os.path.join(os.path.dirname(__file__), "..")
    model_file = os.path.join(repository_root, "integration-tests", "test_user_workflow_spec2vec.model")
    return gensim.models.Word2Vec.load(model_file)
