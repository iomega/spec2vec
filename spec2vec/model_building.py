from typing import List
from typing import Tuple
from typing import Union
import gensim
from spec2vec.utils import ModelSaver
from spec2vec.utils import TrainingProgressLogger


def train_new_word2vec_model(documents: List, iterations: Union[List, int], filename: str = None, **kwargs):
    """Train a new Word2Vec model (using gensim). Save to file if filename is given.

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
        using the name: file_model_word2ve + "_TEMP_#epoch.model".
    filename: str,
        Filename to save model. Default is None, which means no model will be saved.
    sg: int (0,1)
        For sg = 0 --> CBOW model, for sg = 1 --> skip gram model
        (see Gensim documentation). Default for Spec2Vec is 0.
    negative: int
        from Gensim:  If > 0, negative sampling will be used, the int for
        negative specifies how many “noise words” should be drawn (usually
        between 5-20). If set to 0, no negative sampling is used.
        Default for Spec2Vec is 5.
    size: int,
        Dimensions of word vectors Default is 300.
    window: int,
        Window size for context words (small for local context, larger for
        global context). Spec2Vec expects large windwos. Default is 500.
    min_count: int,
        Only consider words that occur at least min_count times in the corpus.
        Default is 1.
    workers: int,
        Number of threads to run the training on (should not be more than
        number of cores/threads. Default is 4.
    learning_rate_initial: float
        Set initial learning rate.
    learning_rate_decay: float
        After every epoch the learning rate will be lowered by the learning_rate_decay.
    progress_logger: bool
        If True, the training progress will be printed every epoch.

    Returns
    -------
    word2vec_model
        Gensim word2vec model.
    """

    settings = {
        "sg": 0,
        "negative": 5,
        "size": 300,
        "window": 500,
        "min_count": 1,
        "workers": 4,
        "learning_rate_initial": 0.025,
        "learning_rate_decay": 0.00025,
        "progress_logger": True,
    }

    # Replace default parameters with input
    for key, value in kwargs.items():
        if key in settings:
            print("The value of {} is set from {} (default) to {}".format(key, settings[key], value))
            settings[key] = value

    num_of_epochs = max(iterations) if isinstance(iterations, list) else iterations
    alpha, min_alpha = set_learning_rate_decay(settings["learning_rate_initial"],
                                               settings["learning_rate_decay"], num_of_epochs)

    # Set callbacks
    callbacks = []
    if settings["progress_logger"]:
        training_progress_logger = TrainingProgressLogger(num_of_epochs)
        callbacks.append(training_progress_logger)
    if filename:
        if isinstance(iterations, int):
            iterations = [iterations]
        model_saver = ModelSaver(num_of_epochs, iterations, filename)
        callbacks.append(model_saver)

    # Train word2vec model
    model = gensim.models.Word2Vec(documents, sg=settings["sg"], negative=settings["negative"],
                                   size=settings["size"], window=settings["window"],
                                   min_count=settings["min_count"], workers=settings["workers"],
                                   iter=num_of_epochs, alpha=alpha, min_alpha=min_alpha,
                                   seed=321, compute_loss=True, callbacks=callbacks)

    return model


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
        print("Warning! Number of total iterations is too high for given learning_rate decay.")
        print("Learning_rate_decay will be set from {} to {}.".format(learning_rate_decay,
                                                                      learning_rate_initial/num_of_epochs))
        min_alpha = 0
    return learning_rate_initial, min_alpha
