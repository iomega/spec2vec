import os
import gensim
import numpy as np
import pytest
from matchms import Spectrum
from spec2vec import SpectrumDocument
from spec2vec.model_building import (set_learning_rate_decay,
                                     train_new_word2vec_model)


def test_set_learning_rate_decay():
    """Test if correct alpha and min_alpha are calculated."""
    alpha, min_alpha = set_learning_rate_decay(0.5, 0.05, 8)
    assert alpha == 0.5, "Expected different alpha."
    assert min_alpha == 0.5 - 8 * 0.05, "Expected different min_alpha"


def test_set_learning_rate_decay_rate_too_high():
    """Test if correct alpha and min_alpha are calculated if rate is too high."""
    alpha, min_alpha = set_learning_rate_decay(0.5, 0.05, 20)
    assert alpha == 0.5, "Expected different alpha."
    assert min_alpha == 0.0, "Expected different min_alpha"


def test_train_new_word2vec_model():
    """Test training of a dummy model."""
    # Create fake corpus
    documents = []
    for i in range(100):
        spectrum = Spectrum(mz=np.linspace(i, 9+i, 10),
                            intensities=np.ones((10)).astype("float"),
                            metadata={})
        documents.append(SpectrumDocument(spectrum, n_decimals=1))
    model = train_new_word2vec_model(documents, iterations=20, vector_size=20,
                                     progress_logger=False)
    assert model.sg == 0, "Expected different default value."
    assert model.negative == 5, "Expected different default value."
    assert model.window == 500, "Expected different default value."
    assert model.alpha == 0.025, "Expected different default value."
    assert model.min_alpha == 0.02, "Expected different default value."
    assert model.epochs == 20, "Expected differnt number of epochs."
    assert model.wv.vector_size == 20, "Expected differnt vector size."
    assert len(model.wv) == 109, "Expected different number of words in vocab."
    assert model.wv.get_vector(documents[0].words[1]).shape[0] == 20, "Expected differnt vector size."


def test_train_new_word2vec_model_with_logger_and_saving(tmp_path):
    """Test training of a dummy model and save it."""
    # Create fake corpus
    documents = []
    for i in range(100):
        spectrum = Spectrum(mz=np.linspace(i, 9+i, 10),
                            intensities=np.ones((10)).astype("float"),
                            metadata={})
        documents.append(SpectrumDocument(spectrum, n_decimals=1))
    # Train model and write to file
    filename = os.path.join(tmp_path, "test.model")
    model = train_new_word2vec_model(documents, iterations=20, filename=filename,
                                     vector_size=20, progress_logger=True)

    # Test if file exists
    assert os.path.isfile(filename), "Could not find saved model file."

    # Test if saved model seems to be correct
    model = gensim.models.Word2Vec.load(filename)
    assert model.sg == 0, "Expected different default value."
    assert model.negative == 5, "Expected different default value."
    assert model.window == 500, "Expected different default value."
    assert model.alpha == 0.025, "Expected different default value."
    assert model.min_alpha == 0.02, "Expected different default value."
    assert model.epochs == 20, "Expected differnt number of epochs."
    assert model.wv.vector_size == 20, "Expected differnt vector size."
    assert len(model.wv) == 109, "Expected different number of words in vocab."
    assert model.wv.get_vector(documents[0].words[1]).shape[0] == 20, "Expected differnt vector size."


def test_train_new_word2vec_model_wrong_entry():
    """Test training of a dummy model with not-accepted gensim argument entry."""
    # Create fake corpus
    documents = []
    for i in range(10):
        spectrum = Spectrum(mz=np.linspace(i, 9+i, 10),
                            intensities=np.ones((10)).astype("float"),
                            metadata={})
        documents.append(SpectrumDocument(spectrum, n_decimals=1))

    with pytest.raises(AssertionError) as msg:
        _ = train_new_word2vec_model(documents, iterations=20, alpha=0.01,
                                     progress_logger=False)

    expected_message_part = "Expect 'learning_rate_initial' instead of 'alpha'."
    assert expected_message_part in str(msg.value), "Expected particular error message."
