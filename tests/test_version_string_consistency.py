import os
import re
from spec2vec import __version__ as expected_version


def test_version_string_consistency():
    """Check whether version in conda/meta.yaml is consistent with that in spec2vec.__version__"""

    repository_root = os.path.join(os.path.dirname(__file__), "..")
    fixture = os.path.join(repository_root, "conda", "meta.yaml")

    with open(fixture, "r", encoding="utf-8") as f:
        metayaml_contents = f.read()

    match = re.search(r"^{% set version = \"(?P<semver>.*)\" %}$", metayaml_contents, re.MULTILINE)
    actual_version = match["semver"]

    assert expected_version == actual_version, "Expected version string used in conda/meta.yaml to be consistent with" \
                                               " that in spec2vec.__version__"
