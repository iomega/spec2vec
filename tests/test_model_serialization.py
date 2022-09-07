from gensim.models import Word2Vec
import json
import os
import pytest
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from spec2vec.serialization.model_exporting import export_model
from spec2vec.serialization.model_importing import import_model, Word2VecLight
from unittest.mock import MagicMock, patch


@pytest.fixture(params=["numpy", "scipy_csr", "scipy_csc"])
def model(request, test_dir):
    model_file = os.path.join(test_dir, "..", "integration-tests", "test_user_workflow_spec2vec.model")
    model = Word2Vec.load(model_file)

    if request.param in ["scipy_csc", "scipy_csr"]:
        scipy_matrix_builder = {"scipy_csr": csr_matrix, "scipy_csc": csc_matrix}
        model.wv.__numpys, model.wv.__ignoreds = [], []
        model.wv.__scipys = ["vectors"]
        model.wv.vectors = scipy_matrix_builder[request.param](model.wv.vectors)
    return model


def write_read_model(model, tmp_path):
    model_file = tmp_path / "model.json"
    weights_file = tmp_path / "weights.npy"
    export_model(model, model_file, weights_file)

    model = import_model(model_file, weights_file)
    return model


def test_write_model_to_disk(model, tmp_path):
    model_file = tmp_path / "model.json"
    weights_file = tmp_path / "weights.npy"
    export_model(model, model_file, weights_file)

    assert os.path.isfile(model_file)
    assert os.path.isfile(weights_file)


def test_read_model_from_disk(test_dir):
    model_file = os.path.join(test_dir, "data", "model.json")
    weights_file = os.path.join(test_dir, "data", "weights.npy")
    model = import_model(model_file, weights_file)

    assert isinstance(model, Word2VecLight)


def test_model_metadata_integrity(model, tmp_path):
    imported_model = write_read_model(model, tmp_path)

    assert imported_model.wv.vector_size == model.wv.vector_size
    assert imported_model.wv.key_to_index == model.wv.key_to_index
    assert imported_model.wv.index_to_key == model.wv.index_to_key
    assert imported_model.wv.__scipys == model.wv.__scipys
    assert imported_model.wv.__numpys == model.wv.__numpys
    assert imported_model.wv.__ignoreds == model.wv.__ignoreds


@pytest.mark.parametrize("model", ["numpy"], indirect=True)
def test_dense_weights_integrity(model, tmp_path):
    imported_model = write_read_model(model, tmp_path)

    assert (imported_model.wv.vectors == model.wv.vectors).all()


@pytest.mark.parametrize("model", ["scipy_csr", "scipy_csc"], indirect=True)
def test_sparse_weights_integrity(model, tmp_path):
    imported_model = write_read_model(model, tmp_path)

    assert (imported_model.wv.vectors.toarray() == model.wv.vectors.toarray()).all()


@patch("json.load", MagicMock(return_value={"unexpected_key": "value", "__weights_format": "np.ndarray"}))
def test_reading_model_with_wrong_keys_fails(test_dir):
    model_file = os.path.join(test_dir, "data", "model.json")
    weights_file = os.path.join(test_dir, "data", "weights.npy")

    with pytest.raises(ValueError) as error:
        import_model(model_file, weights_file)

    assert str(error.value) == "The model dictionary representation does not contain the expected keys."


def test_writing_model_with_wrong_weights_format_fails(model):
    model.wv.vectors = coo_matrix(model.wv.vectors)

    with pytest.raises(NotImplementedError) as error:
        export_model(model, "model.json", "weights.npy")

    assert str(error.value) == "The model's weights format is not supported."
