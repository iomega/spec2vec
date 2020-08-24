import numba
from typing import List
from gensim.models.callbacks import CallbackAny2Vec


class TrainingProgressLogger(CallbackAny2Vec):
    """Callback to log training progress."""

    def __init__(self, num_of_epochs: int):
        """

        Parameters
        ----------
        num_of_epochs:
            Total number of training epochs.
        """
        self.epoch = 0
        self.num_of_epochs = num_of_epochs
        self.loss = 0

    def on_epoch_end(self, model):
        """Return progress of model training"""
        loss = model.get_latest_training_loss()

        print('\r',
              ' Epoch ' + str(self.epoch+1) + ' of ' + str(self.num_of_epochs) + '.',
              end="")
        print('Change in loss after epoch {}: {}'.format(self.epoch+1, loss - self.loss))
        self.epoch += 1
        self.loss = loss


class ModelSaver(CallbackAny2Vec):
    """Callback to save model during training (when specified)."""

    def __init__(self, num_of_epochs: int, iterations: List, filename: str):
        """

        Parameters
        ----------
        num_of_epochs:
            Total number of training epochs.
        iterations:
            Number of total iterations or list of iterations at which to save the
            model.
        filename:
            Filename to save model.
        """
        self.epoch = 0
        self.num_of_epochs = num_of_epochs
        self.iterations = iterations
        self.filename = filename

    def on_epoch_end(self, model):
        """Allow saving model during training when specified in iterations."""
        self.epoch += 1

        if self.filename and self.epoch in self.iterations:
            if self.epoch < self.num_of_epochs:
                filename = self.filename.split(".model")[0] + "_iter_{}.model".format(self.epoch)
            else:
                filename = self.filename
            print("Saving model with name:", filename)
            model.save(filename)


@numba.njit
def cosine_similarity_matrix(vectors_1: numpy.ndarray, vectors_2: numpy.ndarray) -> numpy.ndarray:
    """Fast implementation of cosine similarity between two arrays of vectors.

    Parameters
    ----------
    vectors_1
        Numpy array of vectors. vectors_1.shape[0] is number of vectors, vectors_1.shape[1]
        is vector dimension.
    vectors_2
        Numpy array of vectors. vectors_2.shape[0] is number of vectors, vectors_2.shape[1]
        is vector dimension.
    """
    vectors_1 = vectors_1.copy()
    vectors_2 = vectors_2.copy()
    norm_1 = np.sum(vectors1**2, axis=1) ** (1/2)
    norm_2 = np.sum(vectors2**2, axis=1) ** (1/2)
    for i in range(vectors_1.shape[0]):
        vectors_1[i] = vectors_1[i] / norm_1[i]
    for i in range(vectors_2.shape[0]):
        vectors_2[i] = vectors_2[i] / norm_2[i]
    return np.dot(vectors_1, vectors_2.T)
