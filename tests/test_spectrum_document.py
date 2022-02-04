import numpy as np
import pytest
from matchms import Spectrum
from matchms.filtering import add_losses
from spec2vec import SpectrumDocument


def test_spectrum_document_init_n_decimals_default_value_no_losses():

    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 0.01, 0.1, 1], dtype="float")
    metadata = dict(precursor_mz=100.0)
    spectrum = Spectrum(mz=mz, intensities=intensities, metadata=metadata)
    spectrum_document = SpectrumDocument(spectrum)

    assert spectrum_document.n_decimals == 2, "Expected different default for n_decimals"
    assert len(spectrum_document) == 4
    assert spectrum_document.words == [
        "peak@10.00", "peak@20.00", "peak@30.00", "peak@40.00"
    ]
    assert next(spectrum_document) == "peak@10.00"


def test_spectrum_document_init_n_decimals_1_no_losses():
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 0.01, 0.1, 1], dtype="float")
    metadata = dict(precursor_mz=100.0)
    spectrum = Spectrum(mz=mz, intensities=intensities, metadata=metadata)
    spectrum_document = SpectrumDocument(spectrum, n_decimals=1)

    assert spectrum_document.n_decimals == 1
    assert len(spectrum_document) == 4
    assert spectrum_document.words == [
        "peak@10.0", "peak@20.0", "peak@30.0", "peak@40.0"
    ]
    assert next(spectrum_document) == "peak@10.0"


def test_spectrum_document_init_default_with_losses():
    """Use default n_decimal and add losses."""
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 0.01, 0.1, 1], dtype="float")
    metadata = dict(precursor_mz=100.0)
    spectrum_in = Spectrum(mz=mz, intensities=intensities, metadata=metadata)
    spectrum = add_losses(spectrum_in)
    spectrum_document = SpectrumDocument(spectrum)

    assert spectrum_document.n_decimals == 2, "Expected different default for n_decimals"
    assert len(spectrum_document) == 8
    assert spectrum_document.words == [
        "peak@10.00", "peak@20.00", "peak@30.00", "peak@40.00",
        "loss@60.00", "loss@70.00", "loss@80.00", "loss@90.00"
    ]
    assert next(spectrum_document) == "peak@10.00"


def test_spectrum_document_init_n_decimals_1():
    """Use n_decimal=1 and add losses."""
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 0.01, 0.1, 1], dtype="float")
    metadata = dict(precursor_mz=100.0)
    spectrum_in = Spectrum(mz=mz, intensities=intensities, metadata=metadata)
    spectrum = add_losses(spectrum_in)
    spectrum_document = SpectrumDocument(spectrum, n_decimals=1)

    assert spectrum_document.n_decimals == 1
    assert len(spectrum_document) == 8
    assert spectrum_document.words == [
        "peak@10.0", "peak@20.0", "peak@30.0", "peak@40.0",
        "loss@60.0", "loss@70.0", "loss@80.0", "loss@90.0"
    ]
    assert next(spectrum_document) == "peak@10.0"


def test_spectrum_document_metadata_getter():
    """Test metadata getter"""
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 0.01, 0.1, 1], dtype="float")
    metadata = {"precursor_mz": 100.0,
                "smiles": "testsmiles"}
    spectrum_in = Spectrum(mz=mz, intensities=intensities, metadata=metadata)
    spectrum_document = SpectrumDocument(spectrum_in, n_decimals=2)

    assert spectrum_document.n_decimals == 2
    assert len(spectrum_document) == 4
    assert spectrum_document.metadata == metadata, "Expected different metadata"
    assert spectrum_document.get("smiles") == "testsmiles", "Expected different metadata"
    assert spectrum_document.words == [
        "peak@10.00", "peak@20.00", "peak@30.00", "peak@40.00"
    ]
    assert next(spectrum_document) == "peak@10.00"


def test_spectrum_document_metadata_getter_notallowed_key():
    """Test metadata getter with key that is also attribute"""
    mz = np.array([10], dtype="float")
    intensities = np.array([0], dtype="float")
    metadata = {"smiles": "testsmiles"}
    spectrum_in = Spectrum(mz=mz, intensities=intensities, metadata=metadata)
    spectrum_document = SpectrumDocument(spectrum_in, n_decimals=2)

    assert spectrum_document.n_decimals == 2
    with pytest.raises(AssertionError) as msg:
        spectrum_document.get("n_decimals")

    assert str(msg.value) == "Key cannot be attribute of SpectrumDocument class"


def test_spectrum_document_peak_getter():
    """Test peak getter"""
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 0.01, 0.1, 1], dtype="float")
    metadata = {"precursor_mz": 100.0}
    spectrum_in = Spectrum(mz=mz, intensities=intensities, metadata=metadata)
    spectrum_document = SpectrumDocument(spectrum_in, n_decimals=2)

    assert spectrum_document.words == [
        "peak@10.00", "peak@20.00", "peak@30.00", "peak@40.00"
    ]
    assert np.all(spectrum_document.peaks.mz == mz), "Expected different peak m/z"
    assert np.all(spectrum_document.peaks.intensities == intensities), "Expected different peaks"


def test_spectrum_document_losses_getter():
    """Test losses getter"""
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 0.01, 0.1, 1], dtype="float")
    metadata = {"precursor_mz": 100.0}
    spectrum_in = Spectrum(mz=mz, intensities=intensities, metadata=metadata)
    spectrum = add_losses(spectrum_in)
    spectrum_document = SpectrumDocument(spectrum, n_decimals=2)
    assert np.all(spectrum_document.losses.mz == np.array([60., 70., 80., 90.])), \
        "Expected different losses"
    assert np.all(spectrum_document.losses.intensities == intensities[::-1]), \
        "Expected different losses"
