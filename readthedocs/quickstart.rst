Quickstart
==========

Spec2Vec learns word embeddings from mass-spectral peak and neutral-loss
patterns. Peak positions are converted to tokens such as ``peak@100.39``;
when including losses, tokens such as ``loss@18.01`` are added.

1. Prepare and train
-------------------

The following example uses *normalized* spectra and a model trained on
matching token precision. The input ``references.mgf`` is an example file
provided by the user, not a file bundled with Spec2Vec.

.. code-block:: python

   from matchms.importing import load_from_mgf
   from spec2vec import SpectrumDocument
   from spec2vec.model_building import train_new_word2vec_model

   spectra = list(load_from_mgf("references.mgf"))
   # Preprocess spectra first: normalize intensities and remove unsuitable
   # or empty spectra using the matchms version you have installed.
   documents = [SpectrumDocument(s, n_decimals=2) for s in spectra]

   model = train_new_word2vec_model(
       documents,
       iterations=20,
       filename="references.model",
       workers=2,
   )

For peak-and-loss documents use ``SpectrumDocumentWithLosses`` with a
consistent loss m/z interval. Word2Vec training uses Gensim 4.4 in this
release; existing models remain in the Gensim format.

2. Calculate similarities with matchms 1.0
------------------------------------------

.. code-block:: python

   import numpy as np
   from gensim.models import Word2Vec
   from matchms import calculate_scores
   from spec2vec import Spec2Vec

   # references and queries: processed lists of matchms spectra or documents
   model = Word2Vec.load("references.model")
   similarity = Spec2Vec(model, intensity_weighting_power=0.5)
   scores = calculate_scores(references, queries, similarity)

   # Rows = references; columns = queries.
   matrix = scores.to_array("score")
   top_reference_indices = np.argsort(matrix[:, 0])[::-1][:10]
   top_references = [references[i] for i in top_reference_indices]

3. For matchms 0.x
------------------

With matchms 0.x, use the legacy ``Scores`` interface:

.. code-block:: python

   scores = calculate_scores(references, queries, similarity)
   best_matches = scores.scores_by_query(queries[0], sort=True)[:10]

The compatibility release adapts its output according to the installed
matchms API. For either version, make sure the Word2Vec vocabulary uses the
same decimal rounding as the generated spectrum documents.
