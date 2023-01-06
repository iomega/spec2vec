import os
import gensim
import numpy as np
from matchms import calculate_scores
from matchms.filtering import (add_losses, add_parent_mass, default_filters,
                               normalize_intensities,
                               reduce_to_number_of_peaks,
                               require_minimum_number_of_peaks, select_by_mz)
from matchms.importing import load_from_mgf
from spec2vec import Spec2Vec, SpectrumDocument


def test_user_workflow_spec2vec():
    """Test typical user workflow to get from mass spectra to spec2vec similarities.

    This test will run a typical workflow example using a small dataset and a
    pretrained word2vec model. One main aspect of this is to test if users will
    get exactly the same spec2vec similarity scores when starting from a word2vec
    model that was trained and saved elsewhere.
    """
    def apply_my_filters(s):
        """This is how a user would typically design his own pre- and post-
        processing pipeline."""
        s = default_filters(s)
        s = add_parent_mass(s)
        s = normalize_intensities(s)
        s = reduce_to_number_of_peaks(s, n_required=10, ratio_desired=0.5)
        s = select_by_mz(s, mz_from=0, mz_to=1000)
        s = add_losses(s, loss_mz_from=10.0, loss_mz_to=200.0)
        s = require_minimum_number_of_peaks(s, n_required=5)
        return s

    repository_root = os.path.join(os.path.dirname(__file__), "..")
    spectrums_file = os.path.join(repository_root, "tests", "data", "pesticides.mgf")

    # apply my filters to the data
    spectrums = [apply_my_filters(s) for s in load_from_mgf(spectrums_file)]

    # omit spectrums that didn't qualify for analysis
    spectrums = [s for s in spectrums if s is not None]

    # convert spectrums to spectrum 'documents'
    documents = [SpectrumDocument(s, n_decimals=1) for s in spectrums]

    model_file = os.path.join(repository_root, "integration-tests", "test_user_workflow_spec2vec.model")
    if os.path.isfile(model_file):
        model = gensim.models.Word2Vec.load(model_file)
    else:
        # create and train model
        model = gensim.models.Word2Vec([d.words for d in documents], size=5, min_count=1)
        model.train([d.words for d in documents], total_examples=len(documents), epochs=20)
        model.save(model_file)

    # define similarity_function
    spec2vec = Spec2Vec(model=model, intensity_weighting_power=0.5)

    references = documents[:26]
    queries = documents[25:]

    # calculate scores on all combinations of references and queries
    scores = list(calculate_scores(references, queries, spec2vec))

    # filter out self-comparisons
    filtered = [(reference, query, score) for (reference, query, score) in scores if reference != query]

    sorted_by_score = sorted(filtered, key=lambda elem: elem[2], reverse=True)

    actual_top10 = sorted_by_score[:10]

    expected_top10 = [
        (documents[19], documents[25], 0.9999121928249473),
        (documents[20], documents[25], 0.9998846890269892),
        (documents[20], documents[45], 0.9998756073673759),
        (documents[25], documents[45], 0.9998750427994474),
        (documents[19], documents[27], 0.9998722768460854),
        (documents[22], documents[27], 0.9998633023352553),
        (documents[18], documents[27], 0.9998616961532616),
        (documents[19], documents[45], 0.9998528723697396),
        (documents[14], documents[71], 0.9998404364805897),
        (documents[20], documents[27], 0.9998336807761137)
    ]

    assert [x[0] for x in actual_top10] == [x[0] for x in expected_top10]
    assert [x[1] for x in actual_top10] == [x[1] for x in expected_top10]
    assert np.allclose([x[2][0] for x in actual_top10], [x[2] for x in expected_top10]), "Expected different top 10 table."
