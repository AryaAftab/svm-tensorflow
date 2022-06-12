import sys
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from qtdm import tqdm

import math
import tensorflow as tf
from tensorflow.keras import callbacks


class ShowProgress(callbacks.Callback):
    def __init__(self, epochs, step_show=1, metric="accuracy"):
        super(ShowProgress, self).__init__()
        self.epochs = epochs
        self.step_show = step_show
        self.metric = metric

    def on_train_begin(self, logs=None):
        self.pbar = tqdm(range(self.epochs))

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.step_show == 0:

            self.pbar.set_description(f"""Epoch : {epoch + 1} / {self.epochs}, 
            Train {self.metric} : {round(logs[self.metric], 4)}, 
            Valid {self.metric} : {round(logs['val_' + self.metric], 4)}""")

            self.pbar.update(self.step_show)

            
class BestModelWeights(callbacks.Callback):
    def __init__(self, metric="val_accuracy", metric_type="max"):
        super(BestModelWeights, self).__init__()
        self.metric = metric
        self.metric_type = metric_type
        if self.metric_type not in ["min", "max"]:
                raise NameError('metric_type must be min or max')

    def on_train_begin(self, logs=None):
        if self.metric_type == "min":
            self.best_metric = math.inf
        else:
            self.best_metric = -math.inf
        self.best_epoch = 0
        self.model_best_weights = None
        
    def on_epoch_end(self, epoch, logs=None):
        if self.metric_type == "min":
            if self.best_metric >= logs[self.metric]:
                self.model_best_weights = self.model.get_weights()
                self.best_metric = logs[self.metric]
                self.best_epoch = epoch
        else:
            if self.best_metric <= logs[self.metric]:
                self.model_best_weights = self.model.get_weights()
                self.best_metric = logs[self.metric]
                self.best_epoch = epoch

    def on_train_end(self, logs=None):
        self.model.set_weights(self.model_best_weights)
        print(f"\nBest weights is set, Best Epoch was : {self.best_epoch+1}\n")