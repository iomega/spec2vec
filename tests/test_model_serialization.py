from gensim.models import Word2Vec
import os
import pytest
from spec2vec.serialization.model_exporting import export_model
from spec2vec.serialization.model_importing import import_model, Word2VecLight


@pytest.fixture
def model():
    model_file = os.path.join(os.getcwd(), "..", "integration-tests", "test_user_workflow_spec2vec.model")
    model = Word2Vec.load(model_file)
    yield model


def test_write_model_to_disk(model, tmp_path):
    model_file = tmp_path / "model.json"
    weights_file = tmp_path / "weights.npy"
    export_model(model, model_file, weights_file)

    assert os.path.isfile(model_file)
    assert os.path.isfile(weights_file)


def test_read_model_from_disk():
    model_file = os.path.join("data", "model.json")
    weights_file = os.path.join("data", "weights.npy")
    model = import_model(model_file, weights_file)

    assert isinstance(model, Word2VecLight)
