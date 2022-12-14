import numpy as np
import json
from sklearn.model_selection import train_test_split
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

JSON_PATH = "C:/Users/Nico/Documents/Python Projects/APR/data.json"

# mfcc parameters
n_mfcc = 13
n_fft = 2048
hop_length = 512
max_length = 47600
n_frames = math.ceil((max_length / n_fft) * (n_fft / hop_length))

# load data from json file
with open(JSON_PATH, 'r') as fp:
    data = json.load(fp)

mfccs = np.array(data['mfccs'])
labels = data['labels']

# create train-test partitions
inputs_train, inputs_test, targets_train, targets_test = train_test_split(mfccs, labels, test_size=0.2)

targets_train = np.array(targets_train)
targets_test = np.array(targets_test)

# reshape the input from 2-D to 1-D
inputs_train_new = np.empty((len(inputs_train), n_frames * n_mfcc)) 
inputs_test_new = np.empty((len(inputs_test), n_frames * n_mfcc)) 

i = 0
for e in inputs_train:
    inputs_train_new[i] = np.ravel(inputs_train[i])
    i += 1

i = 0
for e in inputs_test:
    inputs_test_new[i] = np.ravel(inputs_test[i])
    i += 1

# create a Decision Tree classification model with sklearn

model = DecisionTreeClassifier(random_state=0)  # build the model

model.fit(inputs_train_new, targets_train)  # train the model

pred_y = model.predict(inputs_test_new)  # predict

# print results
cm = confusion_matrix(targets_test, pred_y)
acc = accuracy_score(targets_test, pred_y)
print("Confusion Matrix:")
print(cm)
print("Accuracy Score: {:.2f}".format(acc))