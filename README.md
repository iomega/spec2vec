    # Load spectra from MGF

[![bio.tools](https://img.shields.io/badge/bio.tools-spec2vec-blue.svg?labelColor=gray&logo=data:image/svg%2bxml;base64,PHN2ZyBpZD0idXVpZC02ZTIxYTIzOC04NWFmLTRlNDctYjI1OC05ZTEyZDg2MzJmYmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDQzMi44NCA0MzIuODQiPjxwYXRoIGQ9Ik01NS4zLDQyNy4yN2wtNDkuNzMtNDkuNzNjLTcuNDMtNy40My03LjQzLTE5LjQ3LDAtMjYuODlsMTMxLjYzLTEzMS42M2M3LjQzLTcuNDMsMTkuNDctNy40MywyNi44OSwwbDQ5LjczLDQ5LjczYzcuNDMsNy40Myw3LjQzLDE5LjQ3LDAsMjYuODlsLTEzMS42MywxMzEuNjNjLTcuNDMsNy40Mi0xOS40Niw3LjQyLTI2Ljg5LDBaIiBmaWxsPSIjZmZmIi8+PHBhdGggZD0iTTIyNy43MSwyNTUuNzFsLTcuMTgsNy4zNy02LjM0LDYuNC01MS4xMy01MC4xOSw2LjM0LTYuNCw3LjE4LTcuMzdjNi42NC02Ljc2LDE3LjQ1LTYuODYsMjQuMjEtLjIybDI2LjY5LDI2LjJjNi43Nyw2LjY0LDYuODcsMTcuNDUuMjMsMjQuMjFoMFoiIGZpbGw9IiNmZmYiLz48cGF0aCBkPSJNNDMwLjQsMTguNTNsLTE2LjExLTE2LjEiIGZpbGw9IiNmZmYiLz48cGF0aCBkPSJNMzQ4LjQ2LDYzLjM3bC0xMTkuNzQsMTE5LjczLTI0Ljk0LDI0Ljk0LDIxLjAxLDIxLjAxLDI0Ljk0LTI0Ljk0LDExOS43NC0xMTkuNzMiIGZpbGw9IiNmZmYiLz48cGF0aCBkPSJNMzY5LjQ0LDg0LjM4bDI3LjE2LDEuOSwzNi4yNC02NS4zTDQxMS44NSwwbC02NS4zLDM2LjIzLDEuOSwyNy4xNyIgZmlsbD0iI2ZmZiIvPjxwYXRoIGQ9Ik0xMTIuNjEsMTU1LjZjLTI5LjE0LDExLjgzLTYzLjgsNS45NC04Ny40NC0xNy43QzIuOTYsMTE1LjY5LTMuNTgsODMuNzcsNS41NCw1NS44MyIgZmlsbD0iI2ZmZiIvPjxwYXRoIGQ9Ik01Ny4xMiw0LjI2YzI3LjkxLTkuMTUsNTkuODctMi41OCw4Mi4wNywxOS42MywyMy42MSwyMy42MSwyOS41Myw1OC4yNSwxNy43LDg3LjM5IiBmaWxsPSIjZmZmIi8+PHBhdGggZD0iTTI3Ny4yMywzMjAuMjJjLTExLjgzLDI5LjE0LTUuOTQsNjMuOCwxNy43LDg3LjQ0LDIyLjIxLDIyLjIxLDU0LjEzLDI4Ljc1LDgyLjA3LDE5LjYzIiBmaWxsPSIjZmZmIi8+PHBhdGggZD0iTTMyMS41NiwyNzUuOTRjMjkuMTQtMTEuODMsNjMuNzgtNS45Miw4Ny4zOSwxNy43LDIyLjIxLDIyLjIxLDI4Ljc4LDU0LjE2LDE5LjYzLDgyLjA3IiBmaWxsPSIjZmZmIi8+PHBhdGggZD0iTTE2My45MSwyMDcuNTRMNDIuNyw5MS4yNmw1MC43MS00OC4xMiwxMzAuMjMsMTM1LjU3LTkuMjIsOS4xOC0xMy4wOSwxMi4yOC01Ljk4LTQuMTloMGMtLjI4LS4xOC00LjMtMi4yMy00LjYtMi4zNi0uMzctLjE2LTEuOTUtMS4xNi01LjctLjg0LTMuMTEuMjctNS43NCwxLjYyLTguNTcsMy45OGwtMTIuNTgsMTAuNzhoLjAxWiIgZmlsbD0iI2ZmZiIvPjxwYXRoIGQ9Ik0yMjQuMTgsMjY4LjI0bDExNS4zMywxMjAuNDQsNDguMzMtNTAuODktMTM0LjUyLTEyOS4zNC05LjIyLDkuMjYtMTIuMzQsMTMuMTMsNC4xNSw1Ljk1aDBjLjE3LjI4LDIuMiw0LjI4LDIuMzMsNC41OC4xNi4zNywxLjE1LDEuOTQuOCw1LjY4LS4yOCwzLjEtMS42NSw1Ljc0LTQuMDMsOC41OGwtMTAuODQsMTIuNjJoLjAxWiIgZmlsbD0iI2ZmZiIvPjwvc3ZnPg==)](https://bio.tools/spec2vec)
[![Bridge](https://img.shields.io/badge/bridge-bio.tools%20%E2%86%92%20github-blue.svg?labelColor=orange&logo=data:image/svg%2bxml;base64,PHN2ZyBpZD0idXVpZC0zYTVmNzA5MS0xMzM2LTQxNGUtYjBmNS1jMDdkMmQwYWM2MDEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDEwLjAxIDYuOCI+PHBhdGggZD0iTTUuNjYsNS42NmwtLjc0Ljc0Yy0uMjYuMjYtLjU5LjQtLjk1LjRzLS43LS4xNC0uOTYtLjRMLjUyLDMuOTFjLS4yMy0uMjMtLjM3LS41My0uMzktLjg1TDAsMS40NWMtLjAzLS4zNy4xLS43NC4zNi0xLjAxQy42NC4xNCwxLjA0LS4wMiwxLjQ1LjAxbDEuNjEuMTJjLjMyLjAzLjYyLjE3Ljg0LjM5bC42Mi42MWMuMTYuMTcuMTYuNDMsMCwuNTktLjE3LjE2LS40My4xNi0uNTksMGwtLjYxLS42MWMtLjA5LS4wOS0uMi0uMTQtLjMzLS4xNWwtMS42LS4xM2MtLjE2LS4wMS0uMzIuMDUtLjQyLjE3LS4xLjEtLjE1LjI1LS4xNC4zOWwuMTMsMS42MWMuMDEuMTIuMDYuMjQuMTUuMzJsMi40OSwyLjVjLjEuMDkuMjMuMTUuMzcuMTUuMTMsMCwuMjYtLjA2LjM2LS4xNWwuNzQtLjc0Yy4wMi4xNC4wOS4yOC4yLjM4LjEuMTEuMjQuMTguMzkuMloiIGZpbGw9IiNmZmYiLz48cGF0aCBkPSJNMTAsMS40NWwtLjEzLDEuNjFjLS4wMi4zMi0uMTYuNjItLjM5Ljg1bC0yLjQ5LDIuNDljLS4yNi4yNi0uNi40LS45Ni40LS4xMiwwLS4yNS0uMDItLjM3LS4wNS0uMjItLjA3LS4zNC0uMy0uMjgtLjUyLjA2LS4yMi4yOS0uMzQuNTEtLjI4LjA1LjAxLjEuMDIuMTQuMDIuMTQsMCwuMjctLjA2LjM3LS4xNWwyLjQ5LTIuNWMuMDktLjA4LjE1LS4yLjE1LS4zMmwuMTMtMS42MWMuMDEtLjE0LS4wNC0uMjktLjE0LS4zOS0uMS0uMTItLjI2LS4xOC0uNDItLjE3bC0xLjYuMTNjLS4xMy4wMS0uMjQuMDYtLjMzLjE1bC0xLjc3LDEuNzdjLS4wMy0uMTQtLjEtLjI3LS4yMS0uMzgtLjEtLjExLS4yNC0uMTgtLjM4LS4ybDEuNzgtMS43OGMuMjItLjIyLjUyLS4zNi44NC0uMzlsMS42MS0uMTJzLjA3LS4wMS4xLS4wMWMuMzgsMCwuNzQuMTYuOTkuNDQuMjYuMjcuMzkuNjQuMzYsMS4wMVoiIGZpbGw9IiNmZmYiLz48Y2lyY2xlIGN4PSI4LjA1IiBjeT0iMS45NiIgcj0iLjcxIiBmaWxsPSIjZmZmIi8+PGNpcmNsZSBjeD0iMS45NiIgY3k9IjEuOTYiIHI9Ii43MSIgZmlsbD0iI2ZmZiIvPjxwYXRoIGQ9Ik00LjUyLDUuMjVjLS4xMi4xMi0uMzEuMTUtLjQ2LjA5LS4wNS0uMDItLjA5LS4wNS0uMTMtLjA5bC0uMzMtLjMzYy0uMjUtLjI1LS4zOS0uNTktLjM5LS45NXMuMTQtLjcuMzktLjk1bC4zMS0uMzFjLjE2LS4xNi40Mi0uMTYuNTgsMCwuMTUuMTUuMTcuMzkuMDMuNTZsLS4zMy4zM2MtLjEuMS0uMTYuMjMtLjE2LjM3cy4wNi4yNy4xNi4zN2wuMzMuMzNjLjE2LjE3LjE2LjQyLDAsLjU4WiIgZmlsbD0iI2ZmZiIvPjxwYXRoIGQ9Ik02Ljc5LDMuOTdjMCwuMzYtLjE0LjctLjM5Ljk1bC0uMzMuMzNjLS4xNy4xNi0uNDMuMTYtLjU5LDAtLjE1LS4xNS0uMTYtLjM5LS4wMy0uNTVsLjM2LS4zNmMuMS0uMS4xNi0uMjMuMTYtLjM3cy0uMDYtLjI3LS4xNi0uMzdsLS4zLS4zYy0uMTYtLjE2LS4xNi0uNDIsMC0uNTkuMDgtLjA4LjE5LS4xMi4yOS0uMTIuMDcsMCwuMTQuMDIuMi4wNXMuMTEuMDguMTYuMTNjLjA4LjA4LjE1LjE1LjI0LjI0LjI1LjI1LjM5LjU5LjM5Ljk1WiIgZmlsbD0iI2ZmZiIvPjwvc3ZnPg==)](https://bio-tools.github.io/biohackathon2025/)

`fair-software.nl <https://fair-software.nl/>`_ recommendations:

|GitHub Badge|
|License Badge|
|Conda Badge| |Pypi Badge| |Research Software Directory Badge|
|Zenodo Badge|
|CII Best Practices Badge| |Howfairis Badge|

Code quality checks:

|GitHub Workflow Status|
|ReadTheDocs Badge|
|Sonarcloud Quality Gate Badge| |Sonarcloud Coverage Badge|

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

.. |Sonarcloud Quality Gate Badge| image:: https://sonarcloud.io/api/project_badges/measure?project=iomega_spec2vec&metric=alert_status
   :target: https://sonarcloud.io/dashboard?id=iomega_spec2vec
   :alt: Sonarcloud Quality Gate

.. |Sonarcloud Coverage Badge| image:: https://sonarcloud.io/api/project_badges/measure?project=iomega_spec2vec&metric=coverage
   :target: https://sonarcloud.io/component_measures?id=iomega_spec2vec&metric=Coverage&view=list
   :alt: Sonarcloud Coverage

.. |GitHub Workflow Status| image:: https://img.shields.io/github/actions/workflow/status/matchms/spec2vec/CI_build.yml?branch=master
   :target: https://img.shields.io/github/workflow/status/iomega/spec2vec/CI%20Build
   :alt: GitHub Workflow Status


***********************
Documentation for users
***********************
For more extensive documentation `see our readthedocs <https://spec2vec.readthedocs.io/en/latest/>`_ or get started with our `spec2vec introduction tutorial <https://blog.esciencecenter.nl/build-a-mass-spectrometry-analysis-pipeline-in-python-using-matchms-part-ii-spec2vec-8aa639571018>`_.

Versions
========
Since version `0.9.0` Spec2Vec uses `gensim >= 4.4.0` which should make it faster and more future proof. Model trained with older versions should still be importable without any issues. If you had scripts that used additional gensim code, however, those might occationally need some adaptation, see also the `gensim documentation on how to migrate your code <https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4>`_.


Installation
============


Prerequisites:  

- Python 3.10, 3.11, 3.12 or 3.13  
- Recommended: Anaconda

We recommend installing spec2vec from Anaconda Cloud with

.. code-block:: console

  conda create --name spec2vec python=3.13
  conda activate spec2vec
  conda install --channel bioconda --channel conda-forge spec2vec

Alternatively, spec2vec can also be installed using ``pip``. When using spec2vec together with ``matchms`` it is important to note that only the Anaconda install will make sure that also ``rdkit`` is installed properly, which is requried for a few matchms filter functions (it is not required for any spec2vec related functionalities though).

.. code-block:: console

  pip install spec2vec

Examples
========
Below a code example of how to process a large data set of reference spectra to
train a word2vec model from scratch. Spectra are converted to documents using ``SpectrumDocument`` which converts spectrum peaks into "words" according to their m/z ratio (for instance "peak@100.39"). A new word2vec model can then trained using ``train_new_word2vec_model`` which will set the training parameters to spec2vec defaults unless specified otherwise. Word2Vec models learn from co-occurences of peaks ("words") across many different spectra.
To get a model that can give a meaningful representation of a set of
given spectra it is desirable to train the model on a large and representative
dataset.

.. code-block:: python

    from matchms import SpectrumProcessor
    from matchms.filtering.default_pipelines import DEFAULT_FILTERS
    from matchms.importing import load_from_mgf
    from spec2vec import SpectrumDocument
    from spec2vec.model_building import train_new_word2vec_model


    spectra = list(load_from_mgf("reference_spectrums.mgf"))
    
    # Add some default filters. You can add more filters functions like require min. number of peaks
    processor = SpectrumProcessor(DEFAULT_FILTERS)
    
    # Apply filter pipeline
    spectra_cleaned, _ = processor.process_spectra(spectra)
    spectra_cleaned = [s for s in spectra_cleaned if s is not None]

    # Create spectrum documents
    reference_documents = [SpectrumDocument(s, n_decimals=2) for s in spectra_cleaned]

    # Train your reference model
    model_file = "references.model"
    model = train_new_word2vec_model(reference_documents, iterations=[10, 20, 30], filename=model_file,
                                     workers=2, progress_logger=True)

Once a word2vec model has been trained, spec2vec allows to calculate the similarities
between mass spectra based on this model. In cases where the word2vec model was
trained on data different than the data it is applied for, a number of peaks ("words")
might be unknown to the model (if they weren't part of the training dataset). To
account for those cases it is important to specify the ``allowed_missing_percentage``,
as in the example below.

.. code-block:: python

    import gensim
    from matchms import calculate_scores
    from spec2vec import Spec2Vec

    # query_spectra loaded from files using https://matchms.readthedocs.io/en/latest/api/matchms.importing.load_from_mgf.html
    query_spectra = list(load_from_mgf("query_spectrums.mgf"))
    query_spectra_cleaned, _ = processor.process_spectra(query_spectra)

    # Omit spectra that didn't qualify for analysis
    query_spectra_cleaned = [s for s in query_spectra_cleaned if s is not None]

    # Import pre-trained word2vec model (see code example above)
    model_file = "references.model"
    model = gensim.models.Word2Vec.load(model_file)

    # Define similarity_function
    spec2vec_similarity = Spec2Vec(model=model, intensity_weighting_power=0.5,
                                   allowed_missing_percentage=5.0)

    # Calculate scores on all combinations of reference spectra and queries
    scores = calculate_scores(reference_documents, query_spectra_cleaned, spec2vec_similarity)

    # Find the highest scores for a query spectrum of interest
    best_matches = scores.scores_by_query(query_spectra_cleaned[0], sort=True)[:10]

    # Return highest scores
    print([x[1] for x in best_matches])


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

The conda packaging is handled by a `recipe at Bioconda <https://github.com/bioconda/bioconda-recipes/blob/master/recipes/spec2vec/meta.yaml>`_.

Publishing to PyPI will trigger the creation of a `pull request on the bioconda recipes repository <https://github.com/bioconda/bioconda-recipes/pulls?q=is%3Apr+is%3Aopen+spec2vec>`_
Once the PR is merged the new version of matchms will appear on `https://anaconda.org/bioconda/spec2vec <https://anaconda.org/bioconda/spec2vec>`_ 


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

Copyright (c) 2023, Netherlands eScience Center & Düsseldorf University of Applied Sciences

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
