from pathlib import Path
import pytest


@pytest.fixture(scope="module")
def test_dir(request):
    """Return the directory of the currently running test script."""
    return Path(request.fspath).parent
