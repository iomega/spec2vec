.. spec2vec documentation master file, created by
   sphinx-quickstart on Tue Apr  7 09:16:44 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to spec2vec's documentation!
====================================

Word2Vec based similarity measure of mass spectrometry data.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   API <api/spec2vec.rst>

Installation
============

Prerequisites:  

- Python 3.7  
- Anaconda

Install spec2vec from Anaconda Cloud with

.. code-block:: console

  # install spec2vec in a new virtual environment to avoid dependency clashes
  conda create --name spec2vec python=3.7
  conda activate spec2vec
  conda install --channel nlesc --channel bioconda --channel conda-forge spec2vec

Examples
========

Train a word2vec model
**********************
Below a code example of how to process a large data set of reference spectra to
train a word2vec model from scratch. Spectra are converted to documents using :py:class:`~spec2vec.SpectrumDocument` which converts spectrum peaks into "words" according to their m/z ratio (for instance ``peak@100.39``). A new word2vec model can then trained using :py:func:`~spec2vec.model_building.train_new_word2vec_model` which will set the training parameters to spec2vec defaults unless specified otherwise. Word2Vec models learn from co-occurences of peaks ("words") across many different spectra.
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

Derive spec2vec similarity scores
*********************************
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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
