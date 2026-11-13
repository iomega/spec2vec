Installation
============

Supported Python versions
-------------------------

Spec2Vec supports Python **3.10, 3.11, 3.12 and 3.13** in this release line.

PyPI
----

Use a fresh virtual environment and install Spec2Vec:

.. code-block:: console

   python -m pip install spec2vec

To test the matchms 1.0 prerelease with the compatible Spec2Vec development
branch, explicitly select the prerelease:

.. code-block:: console

   python -m pip install --pre "matchms==1.0.0.dev1"
   python -m pip check

The published Spec2Vec package and development branch may not be at the same
compatibility level. Check the package version if testing before the new
Spec2Vec release is published.

Conda / Bioconda
---------------

.. code-block:: console

   conda create -n spec2vec -c conda-forge -c bioconda python=3.12 spec2vec
   conda activate spec2vec

Conda package availability and versions can lag behind PyPI. A Conda build
also resolves RDKit and other optional chemistry dependencies according to
the selected packages; installing Spec2Vec alone does not mean every optional
matchms feature is available.
