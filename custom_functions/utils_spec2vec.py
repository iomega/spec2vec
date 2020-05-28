import numpy
from gensim.models.callbacks import CallbackAny2Vec


class EpochLogger(CallbackAny2Vec):
    """Callback to log information about training progress.
    Used to keep track of gensim model training"""

    def __init__(self, num_of_epochs):
        self.epoch = 0
        self.num_of_epochs = num_of_epochs
        self.loss = 0

    def on_epoch_end(self, model):
        """Return progress of model training"""
        loss = model.get_latest_training_loss()
        # loss_now = loss - self.loss_to_be_subed
        print('\r',
              ' Epoch ' + str(self.epoch+1) + ' of ' + str(self.num_of_epochs) + '.',
              end="")
        print('Change in loss after epoch {}: {}'.format(self.epoch+1, loss - self.loss))
        self.epoch += 1
        self.loss = loss


class ModelSaver(CallbackAny2Vec):
    """Callback to save model during training (when specified)."""

    def __init__(self, num_of_epochs, iterations, filename):
        self.epoch = 0
        self.num_of_epochs = num_of_epochs
        self.iterations = iterations
        self.filename = filename

    def on_epoch_end(self, model):
        """Allow saving model during training when specified in iterations."""
        self.epoch += 1

        # Save model during training if specified in iterations list
        if self.filename is not None:
            if self.epoch in [int(x + numpy.sum(self.iterations[:i])) for i, x in enumerate(self.iterations)]:
                # if self.epoch < self.num_of_epochs:
                filename = self.filename.split('.model')[0] + '_iter_' + str(self.epoch) + '.model'
                print('Saving model with name:', filename)
                model.save(filename)
