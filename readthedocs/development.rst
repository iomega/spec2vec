Development
===========

This release uses **Poetry and Prospector**. A future transition to uv and
Ruff is planned, but is not part of the current compatibility release.

.. code-block:: console

   git clone https://github.com/iomega/spec2vec.git
   cd spec2vec
   python -m pip install poetry
   poetry install
   poetry run pytest
   poetry run prospector -o grouped -o pylint:pylint-report.txt

Compatibility testing
---------------------

Continuous integration tests matchms 0.x and ``1.0.0.dev1`` separately. It
should check ``Spec2Vec.pair()``, ``matrix()``, model serialization, and the
matrix-oriented matchms 1.0 score representation.

Documentation builds
--------------------

Install the documentation dependencies and Spec2Vec in an environment:

.. code-block:: console

   python -m pip install -r readthedocs/requirements.txt
   python -m pip install -e .
   python -m sphinx -n -b html readthedocs readthedocs/_build/html

From a Poetry-managed environment that has the documentation dependencies,
replace ``python -m sphinx`` with ``poetry run sphinx-build``.
