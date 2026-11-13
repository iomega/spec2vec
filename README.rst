`fair-software.nl <https://fair-software.nl/>`_ recommendations:

|GitHub Badge|
|License Badge|
|Conda Badge| |Pypi Badge| |Research Software Directory Badge|
|Zenodo Badge|
|CII Best Practices Badge| |Howfairis Badge|

Code quality checks:

|GitHub Workflow Status|
|ReadTheDocs Badge|

.. image:: readthedocs/spec2vec_logo_01.png
   :width: 300px
   :target: readthedocs/spec2vec_logo_01.png
   :align: left
   :alt: spec2vec logo


################################################################################
spec2vec
################################################################################
**Spec2vec** is a novel spectral similarity score inspired by a natural language processing
algorithm -- Word2Vec. Where Word2Vec learns relationships between words in sentences,
**spec2vec** does so for mass fragments and neutral losses in MS/MS spectra.
The spectral similarity score is based on spectral embeddings learnt
from the fragmental relationships within a large set of spectral data.

If you use **spec2vec** for your research, please cite the following references:

Huber F, Ridder L, Verhoeven S, Spaaks JH, Diblen F, Rogers S, van der Hooft JJJ, (2021) "Spec2Vec: Improved mass spectral similarity scoring through learning of structural relationships". PLoS Comput Biol 17(2): e1008724. `doi:10.1371/journal.pcbi.1008724 <https://doi.org/10.1371/journal.pcbi.1008724>`_

(and if you use **matchms** as well:
F. Huber, S. Verhoeven, C. Meijer, H. Spreeuw, E. M. Villanueva Castilla, C. Geng, J.J.J. van der Hooft, S. Rogers, A. Belloum, F. Diblen, J.H. Spaaks, (2020). "matchms - processing and similarity evaluation of mass spectrometry data". Journal of Open Source Software, 5(52), 2411, https://doi.org/10.21105/joss.02411 )

Thanks!



.. |GitHub Badge| image:: https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue
   :target: https://github.com/iomega/spec2vec
   :alt: GitHub Badge

.. |License Badge| image:: https://img.shields.io/github/license/iomega/spec2vec
   :target: https://github.com/iomega/spec2vec
   :alt: License Badge

.. |Conda Badge| image:: https://img.shields.io/conda/v/bioconda/spec2vec?color=blue
   :target: https://bioconda.github.io/recipes/spec2vec/README.html
   :alt: Conda Badge (Bioconda)

.. |Pypi Badge| image:: https://img.shields.io/pypi/v/spec2vec?color=blue
   :target: https://pypi.org/project/spec2vec/
   :alt: spec2vec on PyPI

.. |Research Software Directory Badge| image:: https://img.shields.io/badge/rsd-spec2vec-00a3e3.svg
   :target: https://www.research-software.nl/software/spec2vec
   :alt: Research Software Directory Badge

.. |Zenodo Badge| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3873169.svg
   :target: https://doi.org/10.5281/zenodo.3873169
   :alt: Zenodo Badge

.. |CII Best Practices Badge| image:: https://bestpractices.coreinfrastructure.org/projects/3967/badge
   :target: https://bestpractices.coreinfrastructure.org/projects/3967
   :alt: CII Best Practices Badge
   
.. |Howfairis Badge| image:: https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green
   :target: https://fair-software.eu
   :alt: Howfairis Badge

.. |ReadTheDocs Badge| image:: https://readthedocs.org/projects/spec2vec/badge/?version=latest
    :alt: Documentation Status
    :target: https://spec2vec.readthedocs.io/en/latest/?badge=latest

.. |GitHub Workflow Status| image:: https://img.shields.io/github/actions/workflow/status/iomega/spec2vec/CI_build.yml?branch=master
   :target: https://github.com/iomega/spec2vec/actions/workflows/CI_build.yml
   :alt: GitHub Workflow Status


***********************
Documentation for users
***********************

See the `full documentation <https://spec2vec.readthedocs.io/en/latest/>`_ for
installation, examples, and the API reference.

Compatibility
=============

The upcoming Spec2Vec 0.10.0 compatibility release supports **matchms 1.0**
(including ``1.0.0.dev1``) while retaining support for matchms 0.x.  It keeps
**Gensim 4.4**, the existing Word2Vec model format, and the Spec2Vec embedding
algorithm. Replacing Gensim is outside the scope of this release.

Spec2Vec currently supports Python **3.10--3.13**. For reproducible workflows,
pin your Spec2Vec, matchms, and Gensim versions.

Installation
============

Install the current published release from PyPI:

.. code-block:: console

   python -m pip install spec2vec

Or use a Conda environment and install the Bioconda package:

.. code-block:: console

   conda create -n spec2vec -c conda-forge -c bioconda python=3.12 spec2vec
   conda activate spec2vec

Bioconda packages can lag behind PyPI releases. To test the matchms 1.0
prerelease, explicitly select it in a compatible environment, once the
Spec2Vec compatibility changes are installed:

.. code-block:: console

   python -m pip install --pre "matchms==1.0.0.dev1"

Check the resulting environment with ``python -m pip check``. Do not assume
that a standard installation will automatically choose prerelease versions.

Quick start
===========

Spec2Vec represents spectra as lists of ``peak@...`` and optionally
``loss@...`` words. A Gensim Word2Vec model learns relationships between
these words; Spec2Vec compares the resulting spectrum embeddings.

The example below assumes you have a reference MGF file. Its intensities
must be normalized to a maximum of 1 before creating spectrum documents.

.. code-block:: python

   from matchms.importing import load_from_mgf
   from spec2vec import SpectrumDocument
   from spec2vec.model_building import train_new_word2vec_model

   references = list(load_from_mgf("references.mgf"))
   documents = [SpectrumDocument(s, n_decimals=2) for s in references]

   model = train_new_word2vec_model(
       documents,
       iterations=20,
       filename="references.model",
       workers=2,
   )

To compare already-preprocessed spectra using a pretrained model:

.. code-block:: python

   import numpy as np
   from gensim.models import Word2Vec
   from matchms import calculate_scores
   from spec2vec import Spec2Vec

   model = Word2Vec.load("references.model")
   similarity = Spec2Vec(model, intensity_weighting_power=0.5)
   scores = calculate_scores(references, queries, similarity)

   # matchms 1.0: rows correspond to references; columns to queries.
   values = scores.to_array("score")
   best_reference_indices = np.argsort(values[:, 0])[::-1][:10]
   best_references = [references[i] for i in best_reference_indices]

Here ``references`` and ``queries`` are lists of preprocessed matchms spectra
or matching spectrum documents. Preprocessing, peak selection, and decimal
rounding must be consistent with the Word2Vec model. See the documentation's
`Quickstart <https://spec2vec.readthedocs.io/en/latest/quickstart.html>`_
for complete loading and preprocessing examples.

**matchms 0.x:** its ``calculate_scores`` result exposes
``scores.scores_by_query(query, sort=True)`` instead of the matrix-oriented
``Scores.to_array("score")`` API.

Glossary of terms
=================

.. list-table::
   :header-rows: 1

   * - Term
     - Description
   * - adduct / addition product
     - During ionization in a mass spectrometer, the molecules of the injected compound break apart
       into fragments. When fragments combine into a new compound, this is known as an addition
       product, or adduct.  `Wikipedia <https://en.wikipedia.org/wiki/Adduct>`__
   * - GNPS
     - Knowledge base for sharing of mass spectrometry data (`link <https://gnps.ucsd.edu/ProteoSAFe/static/gnps-splash.jsp>`__).
   * - InChI / :code:`INCHI`
     - InChI is short for International Chemical Identifier. InChIs are useful
       in retrieving information associated with a certain molecule from a
       database.
   * - InChIKey / InChI key / :code:`INCHIKEY`
     - An identifier for molecules. For example, the InChI key for carbon
       dioxide is :code:`InChIKey=CURLTUGMZLYLDI-UHFFFAOYSA-N` (yes, it
       includes the substring :code:`InChIKey=`).
   * - MGF File / Mascot Generic Format
     - A plain ASCII file format to store peak list data from a mass spectrometry experiment. Links: `matrixscience.com <http://www.matrixscience.com/help/data_file_help.html#GEN>`__,
       `fiehnlab.ucdavis.edu <https://fiehnlab.ucdavis.edu/projects/lipidblast/mgf-files>`__.
   * - parent mass / :code:`parent_mass`
     - Actual mass (in Dalton) of the original compound prior to fragmentation.
       It can be recalculated from the precursor m/z by taking
       into account the charge state and proton/electron masses.
   * - precursor m/z / :code:`precursor_mz`
     - Mass-to-charge ratio of the compound targeted for fragmentation.
   * - SMILES
     - A line notation for describing the structure of chemical species using
       short ASCII strings. For example, water is encoded as :code:`O[H]O`,
       carbon dioxide is encoded as :code:`O=C=O`, etc. SMILES-encoded species may be converted to InChIKey `using a resolver like this one <https://cactus.nci.nih.gov/chemical/structure>`__. The Wikipedia entry for SMILES is `here <https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system>`__.


****************************
Documentation for developers
****************************

Development installation
========================

The current repository uses Poetry for development and CI. For example:

.. code-block:: console

   git clone https://github.com/iomega/spec2vec.git
   cd spec2vec
   python -m pip install --upgrade poetry
   poetry install
   poetry run pytest
   poetry run prospector -o grouped -o pylint:pylint-report.txt

The planned migration to uv and Ruff will be documented separately; the
commands above describe the current tooling.

Documentation build
===================

Install the documentation dependencies described in
``readthedocs/requirements.txt`` and run from the repository root:

.. code-block:: console

   python -m sphinx -b html readthedocs readthedocs/_build/html

For a local build using the Poetry environment, run the equivalent
``poetry run sphinx-build -b html readthedocs readthedocs/_build/html``.

Conda package
=============

The Conda recipe is under ``conda/`` in this repository. The published
Bioconda recipe is at
`bioconda-recipes <https://github.com/bioconda/bioconda-recipes/tree/master/recipes/spec2vec>`_.
Bioconda packages may be updated separately from PyPI releases.

Contributing
============

See `CONTRIBUTING.md <CONTRIBUTING.md>`_ for contributor guidelines.

License
=======

Copyright (c) Florian Huber @ Düsseldorf University of Applied Sciences & Netherlands eScience Center and contributors.

Licensed under the Apache License, Version 2.0. See ``LICENSE`` for details.
