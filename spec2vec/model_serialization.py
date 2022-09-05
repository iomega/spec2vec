import os
from gensim.models import Word2Vec
import json
from numpy import save as save_npy
from scipy.sparse import save_npz
from typing import Union


def export_model(model: Word2Vec,
                 output_model_file: Union[str, os.PathLike],
                 output_weights_file: Union[str, os.PathLike]):
    """Write a lightweight version of Spec2Vec model to disk. Such a model can be read to calculate scores
    but is not fit for further training."""
    keyedvectors_dict = model.wv.__dict__

    # Remove redundant keys
    vectors = keyedvectors_dict.pop('vectors', ValueError('No vectors found in model')) # is save as npy/npz
    keyedvectors_dict.pop('vectors_lockf', None) # another numpy array not needed for scoring

    # From gensim/models/keyedvectors.py https://github.com/RaRe-Technologies/gensim/blob/ded78776284ad7b55b6626191eaa8dcea0dd3db0/gensim/models/keyedvectors.py#L248
    # "expandos" are extra attributes stored for each key: {attribute_name} => numpy array of values of
    # this attribute, with one array value for each vector key.
    # The same information used to be stored in a structure called Vocab in Gensim <4.0.0, but
    # with different indexing: {vector key} => Vocab object containing all attributes for the given vector key.
    #
    # Don't modify expandos directly; call set_vecattr()/get_vecattr() instead.
    keyedvectors_dict.pop('expandos', None)

    # Save model
    with open(output_model_file, 'w') as f:
        json.dump(keyedvectors_dict, f)

    # Save weights
    if keyedvectors_dict['__numpys'] or keyedvectors_dict['__ignoreds']:
        save_npy(output_weights_file, vectors)
    elif keyedvectors_dict['__scipys']:
        save_npz(output_weights_file, vectors)
    else:
        raise AttributeError('The model contains no weights.')
