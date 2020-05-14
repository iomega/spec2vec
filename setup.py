#!/usr/bin/env python

import os

from setuptools import setup
from setuptools import find_packages

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
    long_description=readme + "\n\n",
    author="Netherlands eScience Center",
    author_email="generalization@esciencecenter.nl",
    url="https://github.com/iomega/spec2vec",
    packages=find_packages(),
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords=[
        "python",
        "word2vec",
        "mass spectrometry",
        "fuzzy matching",
        "fuzzy searching",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7"
    ],
    test_suite="tests",
    install_requires=[
        # see environment.yml
    ],
    setup_requires=[
        # see environment.yml
    ],
    tests_require=[
        # see environment.yml
    ],
    extras_require={
        # see environment.yml
    }
)
