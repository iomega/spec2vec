import importlib
import os
import pytest
import tomli


@pytest.mark.skipif(not os.getenv("CI"),reason="Skipping version consistency test outside CI environment")
def test_version_string_consistency():
    """Check whether version in conda/meta.yaml is consistent with that in spec2vec.__version__"""

    repository_root = os.path.join(os.path.dirname(__file__), "..")
    pyproject_file = os.path.join(repository_root, "pyproject.toml")

    with open(pyproject_file, "rb") as f:
        pyproject = tomli.load(f)
    expected_version = pyproject["tool"]["poetry"]["version"]

    spec2vec = importlib.import_module("spec2vec")
    actual_version = spec2vec.__version__

    assert expected_version == actual_version, f"Expected version {expected_version!r} in pyproject.toml to match" \
                                               f" spec2vec.__version__ ({actual_version!r})"
