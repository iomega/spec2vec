################################################################################
spec2vec
################################################################################
Word2Vec based similarity measure of mass spectrometry data.

|

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - fair-software.nl recommendations
     - Badges
   * - \1. Code repository
     - |GitHub Badge|
   * - \2. License
     - |License Badge|
   * - \3. Community Registry
     - |Conda Badge| |Research Software Directory Badge|
   * - \4. Enable Citation
     - |Zenodo Badge|
   * - \5. Checklist
     - |CII Best Practices Badge|
   * - **Other best practices**
     -
   * - Continuous integration
     - |Anaconda Build| |Anaconda Publish|
   * - Documentation
     - |ReadTheDocs Badge|
   * - Code Quality
     - |Sonarcloud Quality Gate Badge| |Sonarcloud Coverage Badge|


.. |GitHub Badge| image:: https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue
   :target: https://github.com/iomega/spec2vec
   :alt: GitHub Badge

.. |License Badge| image:: https://img.shields.io/github/license/iomega/spec2vec
   :target: https://github.com/iomega/spec2vec
   :alt: License Badge

.. |Conda Badge| image:: https://anaconda.org/nlesc/spec2vec/badges/installer/conda.svg
   :target: https://conda.anaconda.org/nlesc
   :alt: Conda Badge
.. |Research Software Directory Badge| image:: https://img.shields.io/badge/rsd-spec2vec-00a3e3.svg
   :target: https://www.research-software.nl/software/spec2vec
   :alt: Research Software Directory Badge

.. |Zenodo Badge| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3873169.svg
   :target: https://doi.org/10.5281/zenodo.3873169
   :alt: Zenodo Badge

.. |CII Best Practices Badge| image:: https://bestpractices.coreinfrastructure.org/projects/3967/badge
   :target: https://bestpractices.coreinfrastructure.org/projects/3967
   :alt: CII Best Practices Badge

.. |ReadTheDocs Badge| image:: https://readthedocs.org/projects/spec2vec/badge/?version=latest
    :alt: Documentation Status
    :scale: 100%
    :target: https://spec2vec.readthedocs.io/en/latest/?badge=latest

.. |Sonarcloud Quality Gate Badge| image:: https://sonarcloud.io/api/project_badges/measure?project=iomega_spec2vec&metric=alert_status
   :target: https://sonarcloud.io/dashboard?id=iomega_spec2vec
   :alt: Sonarcloud Quality Gate

.. |Sonarcloud Coverage Badge| image:: https://sonarcloud.io/api/project_badges/measure?project=iomega_spec2vec&metric=coverage
   :target: https://sonarcloud.io/component_measures?id=iomega_spec2vec&metric=Coverage&view=list
   :alt: Sonarcloud Coverage

.. |Anaconda Build| image:: https://github.com/iomega/spec2vec/workflows/Anaconda%20Build/badge.svg
   :target: https://github.com/iomega/spec2vec/actions?query=workflow%3A%22Anaconda%20Build%22
   :alt: Anaconda Build

.. |Anaconda Publish| image:: https://github.com/iomega/spec2vec/workflows/Anaconda%20Publish/badge.svg
   :target: https://github.com/iomega/spec2vec/actions?query=workflow%3A%22Anaconda%20Publish%22
   :alt: Anaconda Publish

***********************
Documentation for users
***********************
For more extensive documentation `see our readthedocs <https://spec2vec.readthedocs.io/en/latest/>`_.

Installation
============


Prerequisites:  

- Python 3.7  
- Anaconda

Install spec2vec from Anaconda Cloud with

.. code-block:: console

  conda env create --name spec2vec python=3.7
  conda activate spec2vec
  conda install --channel nlesc --channel bioconda --channel conda-forge spec2vec

Examples
========
Below a code example of how to process a large data set of reference spectra to
train a word2vec model from scratch. Spectra are converted to documents using ``SpectrumDocument`` which converts spectrum peaks into "words" according to their m/z ratio (for instance "peak@100.39"). A new word2vec model can then trained using ``train_new_word2vec_model`` which will set the training parameters to spec2vec defaults unless specified otherwise. Word2Vec models learn from co-occurences of peaks ("words") across many different spectra.
To get a model that can give a meaningful representation of a set of
given spectra it is desirable to train the model on a large and representative
dataset.

.. code-block:: python

    import os
    from matchms.filtering import add_losses
    from matchms.filtering import add_parent_mass
    from matchms.filtering import default_filters
    from matchms.filtering import normalize_intensities
    from matchms.filtering import reduce_to_number_of_peaks
    from matchms.filtering import require_minimum_number_of_peaks
    from matchms.filtering import select_by_mz
    from matchms.importing import load_from_mgf
    from spec2vec import SpectrumDocument
    from spec2vec.model_building import train_new_word2vec_model

    def apply_my_filters(s):
        """This is how one would typically design a desired pre- and post-
        processing pipeline."""
        s = default_filters(s)
        s = add_parent_mass(s)
        s = normalize_intensities(s)
        s = reduce_to_number_of_peaks(s, n_required=10, ratio_desired=0.5)
        s = select_by_mz(s, mz_from=0, mz_to=1000)
        s = add_losses(s, loss_mz_from=10.0, loss_mz_to=200.0)
        s = require_minimum_number_of_peaks(s, n_required=10)
        return s

    # Load data from MGF file and apply filters
    spectrums = [spectrum_processing(s) for s in load_from_mgf("reference_spectrums.mgf")]

    # Omit spectrums that didn't qualify for analysis
    spectrums = [s for s in spectrums if s is not None]

    # Create spectrum documents
    reference_documents = [SpectrumDocument(s) for s in spectrums]

    model_file = "references.model"
    model = train_new_word2vec_model(reference_documents, model_file, iterations=[10, 20, 30],
                                     workers=2, progress_logger=True)

Once a word2vec model has been trained, spec2vec allows to calculate the similarities
between mass spectrums based on this model. In cases where the word2vec model was
trained on data different than the data it is applied for, a number of peaks ("words")
might be unknown to the model (if they weren't part of the training dataset). To
account for those cases it is important to specify the ``allowed_missing_percentage``,
as in the example below.

.. code-block:: python

    import gensim
    from matchms import calculate_scores_parallel
    from spec2vec import Spec2VecParallel

    # query_spectrums loaded from files using https://matchms.readthedocs.io/en/latest/api/matchms.importing.load_from_mgf.html
    query_spectrums = [spectrum_processing(s) for s in load_from_mgf("query_spectrums.mgf")]

    # Omit spectrums that didn't qualify for analysis
    query_spectrums = [s for s in query_spectrums if s is not None]

    # Create spectrum documents
    query_documents = [SpectrumDocument(s) for s in query_spectrums]

    # Import pre-trained word2vec model (see code example above)
    model_file = "references.model"
    model = gensim.models.Word2Vec.load(model_file)

    # Define similarity_function
    spec2vec = Spec2VecParallel(model=model, intensity_weighting_power=0.5,
                                allowed_missing_percentage=5.0)

    # Calculate scores on all combinations of reference spectrums and queries
    scores = list(calculate_scores_parallel(reference_documents, query_documents, spec2vec))

    # Filter out self-comparisons
    filtered = [(reference, query, score) for (reference, query, score) in scores if reference != query]

    sorted_by_score = sorted(filtered, key=lambda elem: elem[2], reverse=True)


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
     - An indentifier for molecules. For example, the InChI key for carbon
       dioxide is :code:`InChIKey=CURLTUGMZLYLDI-UHFFFAOYSA-N` (yes, it
       includes the substring :code:`InChIKey=`).
   * - MGF File / Mascot Generic Format
     - A plan ASCII file format to store peak list data from a mass spectrometry experiment. Links: `matrixscience.com <http://www.matrixscience.com/help/data_file_help.html#GEN>`__,
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

Installation
============

To install spec2vec, do:

.. code-block:: console

  git clone https://github.com/iomega/spec2vec.git
  cd spec2vec
  conda env create --file conda/environment-dev.yml
  conda activate spec2vec-dev
  pip install --editable .

Run the linter with:

.. code-block:: console

  prospector

Run tests (including coverage) with:

.. code-block:: console

  pytest


Conda package
=============

To build anaconda package locally, do:

.. code-block:: console

  conda deactivate
  conda env create --file conda/environment-build.yml
  conda activate spec2vec-build
  BUILD_FOLDER=/tmp/spec2vec/_build
  rm -rfv $BUILD_FOLDER;mkdir -p $BUILD_FOLDER
  conda build --numpy 1.18.1 --no-include-recipe -c bioconda -c conda-forge \
  --croot $BUILD_FOLDER ./conda

If successful, this will yield the built ``spec2vec`` conda package as
``spec2vec-<version>*.tar.bz2`` in ``$BUILD_FOLDER/noarch/``. You can test if
installation of this conda package works with:

.. code-block:: console

  # make a clean environment
  conda deactivate
  cd $(mktemp -d)
  conda env create --name test python=3.7
  conda activate test

  conda install \
    --channel bioconda \
    --channel conda-forge \
    --channel file://${CONDA_PREFIX}/noarch/ \
    spec2vec

To publish the package on anaconda cloud, do:

.. code-block:: console

  anaconda --token ${{ secrets.ANACONDA_TOKEN }} upload --user nlesc --force $BUILD_FOLDER/noarch/*.tar.bz2

where ``secrets.ANACONDA_TOKEN`` is a token to be generated on the Anaconda Cloud website. This secret should be added to GitHub repository.


To remove spec2vec package from the active environment:

.. code-block:: console

  conda remove spec2vec


To remove spec2vec environment:

.. code-block:: console

  conda env remove --name spec2vec

Contributing
============

If you want to contribute to the development of spec2vec,
have a look at the `contribution guidelines <CONTRIBUTING.md>`_.

*******
License
*******

Copyright (c) 2020, Netherlands eScience Center

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*******
Credits
*******

This package was created with `Cookiecutter
<https://github.com/audreyr/cookiecutter>`_ and the `NLeSC/python-template
<https://github.com/NLeSC/python-template>`_.
