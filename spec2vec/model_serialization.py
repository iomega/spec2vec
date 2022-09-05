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
    but is not capable of further training.

    Parameters
    ----------
    model:
        Word2Vec trained model.
    output_model_file:
        A path to model json file.
    output_weights_file:
        A path to model's weights file.
    """
    keyedvectors_dict = model.wv.__dict__
    keyedvectors_dict = remove_redundant_keys(keyedvectors_dict)

    weights = keyedvectors_dict.pop('vectors', ValueError('The model contains no weights.'))

    # Save model
    with open(output_model_file, 'w') as f:
        json.dump(keyedvectors_dict, f)

    # Save weights
    if keyedvectors_dict['__numpys'] or keyedvectors_dict['__ignoreds']:
        save_npy(output_weights_file, weights)
    elif keyedvectors_dict['__scipys']:
        save_npz(output_weights_file, weights)
    else:
        raise AttributeError('The model contains no weights.')


def remove_redundant_keys(keyedvectors: dict) -> dict:
    keyedvectors.pop('vectors_lockf', None)
    keyedvectors.pop('expandos', None)
    return keyedvectors
