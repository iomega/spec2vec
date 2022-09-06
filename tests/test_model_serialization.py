from gensim.models import Word2Vec
import os
import pytest
from scipy.sparse import csc_matrix, csr_matrix
from spec2vec.serialization.model_exporting import export_model
from spec2vec.serialization.model_importing import import_model, Word2VecLight


@pytest.fixture(params=["numpy", "scipy_csr", "scipy_csc"])
def model(request):
    model_file = os.path.join(os.getcwd(), "..", "integration-tests", "test_user_workflow_spec2vec.model")
    model = Word2Vec.load(model_file)

    if request.param == "scipy_csc":
        model.wv.__numpys, model.wv.__ignoreds = [], []
        model.wv.__scipys = ["vectors"]
        model.wv.vectors = csc_matrix(model.wv.vectors)
    elif request.param == "scipy_csr":
        model.wv.__numpys, model.wv.__ignoreds = [], []
        model.wv.__scipys = ["vectors"]
        model.wv.vectors = csr_matrix(model.wv.vectors)
    yield model


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


def test_read_model_from_disk():
    model_file = os.path.join("data", "model.json")
    weights_file = os.path.join("data", "weights.npy")
    model = import_model(model_file, weights_file)

    assert isinstance(model, Word2VecLight)


def test_model_metadata_integrity(model, tmp_path):
    imported_model = write_read_model(model, tmp_path)

    assert imported_model.wv.vector_size == model.wv.vector_size
    assert imported_model.wv.key_to_index == model.wv.key_to_index
    assert imported_model.wv.index_to_key == model.wv.index_to_key


@pytest.mark.parametrize("model", ["numpy"], indirect=True)
def test_dense_weights_integrity(model, tmp_path):
    imported_model = write_read_model(model, tmp_path)

    assert (imported_model.wv.vectors == model.wv.vectors).all()


@pytest.mark.parametrize("model", ["scipy_csr", "scipy_csc"], indirect=True)
def test_sparse_weights_integrity(model, tmp_path):
    imported_model = write_read_model(model, tmp_path)

    assert (imported_model.wv.vectors.toarray() == model.wv.vectors.toarray()).all()
