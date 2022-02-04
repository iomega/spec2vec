#!/usr/bin/env python
import os
from setuptools import find_packages, setup


here = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(here, "spec2vec", "__version__.py")) as f:
    exec(f.read(), version)

with open("README.rst") as readme_file:
    readme = readme_file.read()

setup(
    name="spec2vec",
    version=version["__version__"],
    description="Word2Vec based similarity measure of mass spectrometry data.",
    long_description=readme,
    long_description_content_type="text/x-rst",
    author="Netherlands eScience Center",
    author_email="generalization@esciencecenter.nl",
    url="https://github.com/iomega/spec2vec",
    packages=find_packages(),
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords=[
        "word2vec",
        "mass spectrometry",
        "fuzzy matching",
        "fuzzy search"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    test_suite="tests",
    python_requires='>=3.7',
    install_requires=[
        "gensim >=4.0.0",
        "matchms >=0.11.0",
        "numba >=0.51",
        "numpy",
        "tqdm",
    ],
    extras_require={"dev": ["bump2version",
                            "isort>=4.2.5,<5",
                            "pylint<2.12.0",
                            "prospector[with_pyroma]",
                            "pytest",
                            "pytest-cov",
                            "sphinx>=3.0.0,!=3.2.0,<4.0.0",
                            "sphinx_rtd_theme",
                            "sphinxcontrib-apidoc",
                            "yapf",],
    }
)
