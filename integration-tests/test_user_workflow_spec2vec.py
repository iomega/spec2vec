import os
import gensim
import pytest
from matchms import calculate_scores
from matchms.filtering import add_losses
from matchms.filtering import add_parent_mass
from matchms.filtering import default_filters
from matchms.filtering import normalize_intensities
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_mz
from matchms.filtering import select_by_relative_intensity
from matchms.importing import load_from_mgf
from spec2vec import Spec2Vec
from spec2vec import SpectrumDocument


def test_user_workflow_spec2vec():

    def apply_my_filters(s):
        s = default_filters(s)
        s = add_parent_mass(s)
        s = add_losses(s)
        s = normalize_intensities(s)
        s = select_by_relative_intensity(s, intensity_from=0.01, intensity_to=1.0)
        s = select_by_mz(s, mz_from=0, mz_to=1000)
        s = require_minimum_number_of_peaks(s, n_required=5)
        return s

    repository_root = os.path.join(os.path.dirname(__file__), "..")
    spectrums_file = os.path.join(repository_root, "tests", "pesticides.mgf")

    # apply my filters to the data
    spectrums = [apply_my_filters(s) for s in load_from_mgf(spectrums_file)]

    # omit spectrums that didn't qualify for analysis
    spectrums = [s for s in spectrums if s is not None]

    documents = [SpectrumDocument(s) for s in spectrums]

    model_file = os.path.join(repository_root, "integration-tests", "test_user_workflow_spec2vec.model")
    if os.path.isfile(model_file):
        model = gensim.models.Word2Vec.load(model_file)
    else:
        # create and train model
        model = gensim.models.Word2Vec([d.words for d in documents], size=5, min_count=1)
        model.train([d.words for d in documents], total_examples=len(documents), epochs=20)
        model.save(model_file)

    # define similarity_function
    spec2vec = Spec2Vec(model=model, documents=documents, intensity_weighting_power=0.5)

    references = documents[:26]
    queries = documents[25:]

    # calculate scores on all combinations of references and queries
    scores = list(calculate_scores(references, queries, spec2vec))

    # filter out self-comparisons
    filtered = [(reference, query, score) for (reference, query, score) in scores if reference != query]

    sorted_by_score = sorted(filtered, key=lambda elem: elem[2], reverse=True)

    actual_top10 = sorted_by_score[:10]

    expected_top10 = [
        (documents[16], documents[38], pytest.approx(0.9968146516288907)),
        (documents[16], documents[61], pytest.approx(0.9941841832552322)),
        (documents[10], documents[58], pytest.approx(0.9940519283255305)),
        (documents[8],  documents[26], pytest.approx(0.9926849150218476)),
        (documents[16], documents[55], pytest.approx(0.9917865922946578)),
        (documents[16], documents[62], pytest.approx(0.991290192194624)),
        (documents[16], documents[46], pytest.approx(0.9908729442589261)),
        (documents[2],  documents[37], pytest.approx(0.9904318498241543)),
        (documents[10], documents[60], pytest.approx(0.9902712249648715)),
        (documents[14], documents[61], pytest.approx(0.989827726618046))
    ]

    assert actual_top10 == expected_top10
