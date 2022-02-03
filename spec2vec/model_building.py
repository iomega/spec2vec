"""This module contains functions that will help users to train a word2vec model
through gensim.
"""
import logging
from typing import List, Tuple, Union
import gensim
from spec2vec.utils import ModelSaver, TrainingProgressLogger


logger = logging.getLogger("spec2vec")


def train_new_word2vec_model(documents: List, iterations: Union[List[int], int], filename: str = None,
                             progress_logger: bool = True, **settings) -> gensim.models.Word2Vec:
    """Train a new Word2Vec model (using gensim). Save to file if filename is given.

    Example code on how to train a word2vec model on a corpus (=list of documents)
    that is derived from a given set of spectrums (list of matchms.Spectrum instances):

    .. code-block:: python

        from matchms import SpectrumDocument
        from spec2vec.model_building import train_new_word2vec_model

        documents = [SpectrumDocument(s, n_decimals=1) for s in spectrums]
        model = train_new_word2vec_model(documents, iterations=20, size=200,
                                         workers=1, progress_logger=False)

    Parameters
    ----------
    documents:
        List of documents, each document being a list of words (strings).
    iterations:
        Specifies the number of training interations. This can be done by setting
        iterations to the total number of training epochs (e.g. "iterations=15"),
        or by passing a list of iterations (e.g. "iterations=[5,10,15]") which will
        also led to a total training of max(iterations) epochs, but will save the
        model for every iteration in the list. Temporary models will be saved
        using the name: filename_TEMP_{#iteration}epoch.model".
    filename: str,
        Filename to save model. Default is None, which means no model will be saved.
        If a list of iterations is passed (e.g. "iterations=[5,10,15]"), then
        intermediate models will be saved during training (here after 5, 10
        iterations) using the pattern: filename_TEMP_{#iteration}epoch.model
    learning_rate_initial:
        Set initial learning rate. Default is 0.025.
    learning_rate_decay:
        After every epoch the learning rate will be lowered by the learning_rate_decay.
        Default is 0.00025.
    progress_logger:
        If True, the training progress will be printed every epoch. Default is True.
    **settings
        All other named arguments will be passed to the :py:class:`gensim.models.word2vec.Word2Vec` constructor.
    sg: int (0,1)
        For sg = 0 --> CBOW model, for sg = 1 --> skip gram model
        (see Gensim documentation). Default for Spec2Vec is 0.
    negative: int
        from Gensim:  If > 0, negative sampling will be used, the int for
        negative specifies how many “noise words” should be drawn (usually
        between 5-20). If set to 0, no negative sampling is used.
        Default for Spec2Vec is 5.
    size: int,
        Dimensions of word vectors. Default is 300.
    window: int,
        Window size for context words (small for local context, larger for
        global context). Spec2Vec expects large windwos. Default is 500.
    min_count: int,
        Only consider words that occur at least min_count times in the corpus.
        Default is 1.
    workers: int,
        Number of threads to run the training on (should not be more than
        number of cores/threads. Default is 4.

    Returns
    -------
    word2vec_model
        Gensim word2vec model.
    """
    settings = set_spec2vec_defaults(**settings)

    num_of_epochs = max(iterations) if isinstance(iterations, list) else iterations

    # Convert spec2vec style arguments to gensim style arguments
    settings = learning_rates_to_gensim_style(num_of_epochs, **settings)

    # Set callbacks
    callbacks = []
    if progress_logger:
        training_progress_logger = TrainingProgressLogger(num_of_epochs)
        callbacks.append(training_progress_logger)
    if filename:
        if isinstance(iterations, int):
            iterations = [iterations]
        model_saver = ModelSaver(num_of_epochs, iterations, filename)
        callbacks.append(model_saver)

    # Train word2vec model
    model = gensim.models.Word2Vec(documents, callbacks=callbacks, **settings)

    return model


def set_spec2vec_defaults(**settings):
    """Set spec2vec default argument values"(where no user input is give)"."""
    defaults = {
        "sg": 0,
        "negative": 5,
        "vector_size": 300,
        "window": 500,
        "min_count": 1,
        "learning_rate_initial": 0.025,
        "learning_rate_decay": 0.00025,
        "workers": 4,
        "compute_loss": True,
    }
    assert "min_alpha" not in settings, "Expect 'learning_rate_decay' to describe learning rate decrease."
    assert "alpha" not in settings, "Expect 'learning_rate_initial' instead of 'alpha'."

    # Set default parameters or replace by **settings input
    for key, value in defaults.items():
        if key in settings:
            msg = f"The value of {key} is set from {value} (default) to {settings[key]}"
            logger.info(msg)
        else:
            settings[key] = value
    return settings


def learning_rates_to_gensim_style(num_of_epochs, **settings):
    """Convert "learning_rate_initial" and "learning_rate_decay" to gensim
    "alpha" and "min_alpha"."""
    alpha, min_alpha = set_learning_rate_decay(settings["learning_rate_initial"],
                                               settings["learning_rate_decay"], num_of_epochs)
    settings["alpha"] = alpha
    settings["min_alpha"] = min_alpha
    settings["epochs"] = num_of_epochs

    # Remove non-Gensim arguments from settings
    del settings["learning_rate_initial"]
    del settings["learning_rate_decay"]
    return settings


def set_learning_rate_decay(learning_rate_initial: float, learning_rate_decay: float,
                            num_of_epochs: int) -> Tuple[float, float]:
    """The learning rate in Gensim model training is defined by an initial rate
    (alpha) and a final rate (min_alpha). which can be unintuitive. Here those
    parameters will be set based on the given values for learning_rate_initial,
    num_of_epochs, and learning_rate_decay.

    Parameters
    ----------
    learning_rate_initial:
        Set initial learning rate.
    learning_rate_decay:
        After evert epoch, the learning rate will be lowered by the learning_rate_decay.
    number_of_epochs:
        Total number of epochs for training.

    Returns:
    --------
    alpha:
        Initial learning rate.
    min_alpha:
        Final learning rate.
    """
    min_alpha = learning_rate_initial - num_of_epochs * learning_rate_decay
    if min_alpha < 0:
        msg = ("Number of total iterations is too high for given learning_rate decay.",
               f"Learning_rate_decay will be set from {learning_rate_decay} ",
               "to {learning_rate_initial/num_of_epochs}.")
        logger.warning(msg)
        min_alpha = 0
    return learning_rate_initial, min_alpha
