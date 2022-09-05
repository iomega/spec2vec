import os
from gensim.models import Word2Vec
import json
from numpy import save as save_npy
from scipy.sparse import save_npz
from typing import Union


def export_model(model: Word2Vec,
                 output_model_file: Union[str, os.PathLike],
                 output_weights_file: Union[str, os.PathLike]):
    """Write a lightweight version of a Spec2Vec model to disk. Such a model can be read to calculate scores
    but is not capable of further training.

    Parameters
    ----------
    model:
        :py:class:Word2Vec trained model.
    output_model_file:
        A path of json file to save the model.
    output_weights_file:
        A path of npy file to save the model's weights.
    """
    keyedvectors = extract_keyedvectors(model)
    weights = keyedvectors.pop('vectors', ValueError('The model contains no weights.'))

    save_model(keyedvectors, output_model_file)
    save_weights(keyedvectors, weights, output_weights_file)


def save_weights(keyedvectors, weights, output_weights_file):
    """Write model's weights to disk in npy or npz format."""
    if keyedvectors['__numpys'] or keyedvectors['__ignoreds']:
        save_npy(output_weights_file, weights)
    elif keyedvectors['__scipys']:
        save_npz(output_weights_file, weights)
    else:
        raise AttributeError('The model contains no weights.')


def save_model(keyedvectors, output_model_file):
    """Write model's metadata to disk in json format."""
    with open(output_model_file, 'w') as f:
        json.dump(keyedvectors, f)


def extract_keyedvectors(model: Word2Vec) -> dict:
    """
    Extract :py:class:KeyedVectors object from the model, convert it to a dictionary and remove redundant keys.

    Parameters
    ----------
    model:
        Word2Vec trained model.

    Returns
    -------
    keyedvectors:
        Dictionary representation of :py:class:KeyedVectors without redundant keys.
    """
    keyedvectors = model.wv.__dict__
    keyedvectors.pop('vectors_lockf', None)
    keyedvectors.pop('expandos', None)
    return keyedvectors
