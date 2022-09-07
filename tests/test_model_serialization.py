import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from gensim.models import Word2Vec
from matchms import Spectrum, calculate_scores
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from spec2vec import Spec2Vec
from spec2vec.serialization import Word2VecLight, export_model, import_model


@pytest.fixture(params=["numpy", "scipy_csr", "scipy_csc"])
def model(request, test_dir):
    model_file = os.path.join(test_dir, "..", "integration-tests", "test_user_workflow_spec2vec.model")
    model = Word2Vec.load(model_file)

    if request.param in ["scipy_csc", "scipy_csr"]:
        scipy_matrix_builder = {"scipy_csr": csr_matrix, "scipy_csc": csc_matrix}
        model.wv.__numpys, model.wv.__ignoreds = [], []
        model.wv.__scipys = ["vectors"]  # pylint:disable=protected-access
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
    assert imported_model.wv.__scipys == model.wv.__scipys  # pylint:disable=protected-access
    assert imported_model.wv.__numpys == model.wv.__numpys  # pylint:disable=protected-access
    assert imported_model.wv.__ignoreds == model.wv.__ignoreds  # pylint:disable=protected-access


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

    assert str(error.value) == "The keys of model's dictionary representation do not match the expected keys."


def test_writing_model_with_wrong_weights_format_fails(model):
    model.wv.vectors = coo_matrix(model.wv.vectors)

    with pytest.raises(NotImplementedError) as error:
        export_model(model, "model.json", "weights.npy")

    assert str(error.value) == "The model's weights format is not supported."


@pytest.mark.parametrize("model", ["numpy"], indirect=True)  # calculate_scores supports only numpy arrays
def test_reloaded_model_computes_scores(model, tmp_path):
    spectrum_1 = Spectrum(mz=np.array([100, 150, 200.]),
                          intensities=np.array([0.7, 0.2, 0.1]),
                          metadata={'id': 'spectrum1'})
    spectrum_2 = Spectrum(mz=np.array([100, 140, 190.]),
                          intensities=np.array([0.4, 0.2, 0.1]),
                          metadata={'id': 'spectrum2'})
    spectrum_3 = Spectrum(mz=np.array([110, 140, 180.]),
                          intensities=np.array([0.4, 0.3, 0.1]),
                          metadata={'id': 'spectrum3'})

    queries = [spectrum_1, spectrum_2]
    references = [spectrum_1, spectrum_2, spectrum_3]

    reloaded_model = write_read_model(model, tmp_path)
    spec2vec_reloaded = Spec2Vec(reloaded_model, intensity_weighting_power=0.5)
    spec2vec = Spec2Vec(model, intensity_weighting_power=0.5)

    scores = list(calculate_scores(references, queries, spec2vec))
    scores_reloaded = list(calculate_scores(references, queries, spec2vec_reloaded))

    assert scores == scores_reloaded
