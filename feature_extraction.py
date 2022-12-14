import os
import json
from pickletools import optimize
import random
import librosa
from librosa import display
import matplotlib.pyplot as plt
#import cupy as cp
import numpy as np
import math

PATH = "C:/Users/Nico/Documents/Python Projects/APR/data/"
JSON_PATH = "C:/Users/Nico/Documents/Python Projects/APR/data.json"
LABELS = ['Major','Minor']
SAMPLE_RATE = 22050

# store file names in list
filenames = []
for label in LABELS:
    dir_path = PATH + label
    for file in os.scandir(dir_path):
        filenames.append((dir_path + '/' + file.name, LABELS.index(label))) # save a tuple with file path and label for each file

# show total count of files, and count for each class
print('NUMBER OF FILES')
total_files = len(filenames)
major_files = len([label for _,label in filenames if label == 0])
minor_files = len([label for _,label in filenames if label == 1])
print('Total: ' + str(total_files))
print('Major: ' + str(major_files))
print('Minor: ' + str(minor_files))

# removing excess files of class 'Major' to reach an equal count between the 2 classes 
print('\nRemoving excess files from class \'Major\'...\n')
count_diff = major_files - minor_files
c = 0
while c < count_diff:
    filenames.pop(random.randint(0, major_files - c))
    c += 1

# show new total count of files, and count for each class, which is now equal
print('NUMBER OF FILES')
total_files = len(filenames)
major_files = len([label for _,label in filenames if label == 0])
minor_files = len([label for _,label in filenames if label == 1])
print('Total: ' + str(total_files))
print('Major: ' + str(major_files))
print('Minor: ' + str(minor_files))

# shuffle the data points
random.shuffle(filenames)

# save labels
labels = [label for _,label in filenames]

# open files and save mfccs
n_mfcc = 13
n_fft = 2048
hop_length = 512
max_length = 47600  # longer files will be cut at this point
n_frames = math.ceil((max_length / n_fft) * (n_fft / hop_length))  # calculate the number of mfcc frames to be computed, which is constant 

mfccs = np.empty((len(filenames), n_frames, n_mfcc)) 
i = 0
for filename in filenames:
    mfccs[i] = librosa.feature.mfcc(y=librosa.load(filename[0], sr=SAMPLE_RATE)[0][:max_length], n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc).T
    i += 1

# save data to json file
data = {
    'mfccs': mfccs.tolist(),
    'labels': labels,
    'mapping': LABELS
}

print('\nSaving data to json file...')
with open(JSON_PATH, 'w') as fp:
    json.dump(data, fp, indent=4)

'''
# Tests on a singular sample

sample = 113
signal, sr = librosa.load(filenames[sample][0])

# show waveplot
librosa.display.waveshow(signal, sr=SAMPLE_RATE)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# compute and show fourier transform
fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

left_magnitude = magnitude[:int(len(frequency)/2)]
left_frequency = frequency[:int(len(frequency)/2)]

plt.plot(left_frequency, left_magnitude)
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.show()

# compute and show MFCCs of single sample
n_fft = 2048
hop_length = 512
mfcc = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=3)
print(mfcc)
librosa.display.specshow(mfcc, sr=SAMPLE_RATE, hop_length=hop_length)
plt.xlabel('Time - ' + LABELS[filenames[sample][1]])
plt.ylabel('MFCC')
plt.colorbar()
plt.show()
'''