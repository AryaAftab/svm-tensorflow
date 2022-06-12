## SVM - Tensorflow
An implementation of support vector machine (SVM) in tensorflow 2.x. 

## Install

```bash
git clone https://github.com/AryaAftab/svm-tensorflow.git
cd svm-tensorflow/
python setup.py install
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


# Define Bone, if you want linear svm, you can pass None to SVMTrainer as bone
Bone = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(128, activation="relu"),
])


svm_model = SVMTrainer(num_class=2, bone=Bone)
svm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
                  metrics=["accuracy"])

# Callbacks
epochs = 50
show_progress = ShowProgress(epochs)
best_weight = BestModelWeights()

#Train 
history = svm_model.fit(train_x, train_y,
                        epochs=epochs, validation_data=(valid_x, valid_y),
                        callbacks=[show_progress, best_weight],
                        verbose=0 # When you want to use ShowProgress callback, you should set verbose to zero
                        )
```
### Plot result and boundary:
```python
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


plt.figure(figsize=(20, 10))

plt.subplot(1,2,1)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.ylabel("loss")
plt.xlabel("epoch")

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.ylabel("accuracy")
plt.xlabel("epoch")


plt.figure(figsize=(15, 10))

plt.scatter(class1[:,0], class1[:,1])
plt.scatter(class2[:,0], class2[:,1])

plt.scatter(x[:,0], x[:,1])
```
![1](https://user-images.githubusercontent.com/30603302/173213888-269484bd-091f-42df-ad37-56426683c842.png)

![2](https://user-images.githubusercontent.com/30603302/173213895-56a5f996-e0ec-4987-ba0a-abc17767935c.png)
