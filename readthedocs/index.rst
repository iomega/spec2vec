.. spec2vec documentation master file, created by
   sphinx-quickstart on Tue Apr  7 09:16:44 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to spec2vec's documentation!
===================================

Word2Vec based similarity measure of mass spectrometry data.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   API <api/spec2vec.rst>

Example
=======
Below is a small example of using spec2vec to calculate the similarities between mass Spectrums.

.. code-block:: python

    import gensim
    from spec2vec import Spec2VecParallel
    from spec2vec import SpectrumDocument

    # reference_spectrums & query_spectrums loaded from files using https://matchms.readthedocs.io/en/latest/api/matchms.importing.load_from_mgf.html
    references = [SpectrumDocument(s, n_decimals=2) for s in reference_spectrums]
    queries = [SpectrumDocument(s, n_decimals=2) for s in query_spectrums]

    # Import pre-trained word2vec model (alternative: train new model)
    model_file = "path and filename"
    model = gensim.models.Word2Vec.load(model_file)

    # Define similarity_function
    spec2vec = Spec2VecParallel(model=model, intensity_weighting_power=0.5)

    # Calculate scores on all combinations of references and queries
    scores = list(calculate_scores(references, queries, spec2vec))

    # Filter out self-comparisons
    filtered = [(reference, query, score) for (reference, query, score) in scores if reference != query]

    sorted_by_score = sorted(filtered, key=lambda elem: elem[2], reverse=True)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
