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
        print(f'Change in loss after epoch {self.epoch + 1}: {loss - self.loss}')
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
                filename = f"{self.filename.split('.model')[0]}_iter_{self.epoch}.model"
            else:
                filename = self.filename
            print("Saving model with name:", filename)
            model.save(filename)
