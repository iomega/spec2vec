"""Sphinx configuration for Spec2Vec documentation."""
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import os
import sys

DOCS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

project = "spec2vec"
author = "Netherlands eScience Center and Spec2Vec contributors"
copyright = f"2020-{datetime.now().year}, {author}"

try:
    release = version("spec2vec")
except PackageNotFoundError:
    release = "development"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinxcontrib.apidoc",
]

# Keep the established API-generation extension for this minimal upgrade.
apidoc_module_dir = str(PROJECT_ROOT / "spec2vec")
apidoc_output_dir = str(DOCS_DIR / "api")
apidoc_excluded_paths = ["tests", "readthedocs"]
apidoc_separate_modules = True
apidoc_module_first = True
os.environ["SPHINX_APIDOC_OPTIONS"] = "members,show-inheritance"

autodoc_default_options = {"special-members": "__init__,__call__"}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"] if (DOCS_DIR / "_templates").is_dir() else []
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {"navigation_depth": 3, "collapse_navigation": False}
html_static_path = ["_static"] if (DOCS_DIR / "_static").is_dir() else []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "gensim": ("https://radimrehurek.com/gensim/", None),
    "matchms": ("https://matchms.readthedocs.io/en/latest/", None),
}
