## SVM - Tensorflow
An implementation of support vector machine (SVM) in tensorflow 2.x.

## Install
On your machine:
```bash
git clone https://github.com/AryaAftab/svm-tensorflow.git
cd svm-tensorflow/
python setup.py install
```
On google-colab:
```bash
!pip install git+https://github.com/AryaAftab/svm-tensorflow.git
```

## Usage
### Define and train model:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.datasets import make_moons

import tensorflow as tf
from svm_tensorflow import *


# Define Data
data = make_moons(3000, noise=0.05)
x, y = data
y = tf.one_hot(y, depth=2, on_value=1, off_value=0).numpy()

x, y = shuffle(x, y)

n_train = int(0.8 * len(x))
train_x, train_y = x[:n_train], y[:n_train]
valid_x, valid_y = x[n_train:], y[n_train:]

# Define metrics
METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

# Define Bone, if you want linear svm, you can pass None to SVMTrainer as bone
Bone = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(128, activation="relu"),
])


svm_model = SVMTrainer(num_class=2, bone=Bone)
svm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
                  metrics=METRICS)

# Callbacks
epochs = 200
show_progress = ShowProgress(epochs)
best_weight = BestModelWeights()

# Train
history = svm_model.fit(train_x, train_y,
                        epochs=epochs, validation_data=(valid_x, valid_y),
                        callbacks=[show_progress, best_weight],
                        verbose=0 # When you want to use ShowProgress callback, you should set verbose to zero
                        )
```
### Plot result and boundary:
```python
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Helper function for plot metrics
def plot_metrics(history):
    plt.figure(figsize=(12, 10))
    metrics = ['loss', 'prc', 'accuracy', 'fp', 'precision', "tp", "recall", "tn", "auc", "fn"]

    for n, metric in enumerate(metrics):

        name = metric.replace("_"," ").capitalize()
        plt.subplot(5, 2, n+1)

        plt.plot(history.epoch,
                 history.history[metric],
                 color=colors[0],
                 label='Train')

        plt.plot(history.epoch,
                 history.history['val_'+ metric],
                 color=colors[1],
                 #linestyle="--",
                 label='Val')

        plt.xlabel('Epoch')
        plt.ylabel(name)

        plt.legend();

plot_metrics(history)


plt.figure(figsize=(15, 10))
Min = x.min(axis=0)
Max = x.max(axis=0)

a = np.linspace(Min[0], Max[0], 200)
b = np.linspace(Min[1], Max[1], 200)
xa, xb = np.meshgrid(a, b)

X = np.stack([xa, xb], axis=-1)
X = np.reshape(X, [-1, 2])

bound = svm_model.predict(X)
bound = np.argmax(bound, axis=-1)

class1 = X[bound == 0]
class2 = X[bound == 1]

plt.scatter(class1[:,0], class1[:,1])
plt.scatter(class2[:,0], class2[:,1])

plt.scatter(x[:,0], x[:,1])
```
<div align=center>
<img width=95% src="https://user-images.githubusercontent.com/30603302/184471363-66d571a7-5ff6-4f52-8ace-9ee32560b8ae.png"/>
</div>

<div align=center>
<img width=95% src="https://user-images.githubusercontent.com/30603302/184471244-57160568-c8c0-4f88-8c61-95e622c941e3.png"/>
</div>
