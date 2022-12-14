import numpy as np
import json
from sklearn.model_selection import train_test_split
import keras
import datetime
import tensorflow as tf

JSON_PATH = "C:/Users/Nico/Documents/Python Projects/APR/data.json"

# load data from json file
with open(JSON_PATH, 'r') as fp:
    data = json.load(fp)

mfccs = np.array(data['mfccs'])
labels = data['labels']

# create train-test partitions
inputs_train, inputs_test, targets_train, targets_test = train_test_split(mfccs, labels, test_size=0.2)

targets_train = np.array(targets_train)
targets_test = np.array(targets_test)

# add 4th dimension (for channels)
inputs_train = inputs_train[..., np.newaxis]
inputs_test = inputs_test[..., np.newaxis]

# create a CNN classification model with keras
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(inputs_train.shape[1], inputs_train.shape[2], inputs_train.shape[3])),
    keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
    ])

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# tensorboard setup
log_dir = 'APR/logs/fit/' + datetime.datetime.now().strftime("CNN_%d-%m-%Y_%H-%M-%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# train and validate model
model.fit(x=inputs_train, y=targets_train, validation_data=(inputs_test, targets_test), epochs=50, batch_size=16, callbacks=[tensorboard_callback])