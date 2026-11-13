"""Small compatibility bridge for matchms 0.x and 1.x.

Do not infer the API from package version strings: use the public module
location provided by each matchms API generation instead.
"""

try:
    from matchms.similarity.base_similarity import BaseSimilarity
except ModuleNotFoundError as error:
    # Do not accidentally mask import failures *within* an installed matchms.
    if error.name != "matchms.similarity.base_similarity":
        raise
    from matchms.similarity.BaseSimilarity import BaseSimilarity

    MATCHMS_V1 = False
    Scores = None
else:
    from matchms.scores import Scores

    MATCHMS_V1 = True
