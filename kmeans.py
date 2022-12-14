import numpy as np
import json
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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

# reshape the input from 2-D to 1-D
mfccs_new = np.empty((len(mfccs), n_frames * n_mfcc)) 
i = 0
for e in mfccs:
    mfccs_new[i] = np.ravel(mfccs[i])
    i += 1

# normalize data: z = (x - u) / s where u is the mean and s is the standard deviation
mfcc_norm = StandardScaler().fit_transform(mfccs_new)

# apply PCA
pca = PCA(n_components=0.90)
mfccs_reduced = pca.fit_transform(mfcc_norm)
print("\n+++ PCA +++\nOriginal number of features:", mfcc_norm.shape[1])
print("Reduced number of features:", mfccs_reduced.shape[1])

# apply K-Means
print("\nComputing KMeans on the data...")
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(mfccs_reduced)

# visualize K-Means clusters
filtered_label0 = mfccs_reduced[clusters == 0]
filtered_label1 = mfccs_reduced[clusters == 1]
plt.scatter(filtered_label0[:,0] , filtered_label0[:,1] , color = 'red', label = 0)
plt.scatter(filtered_label1[:,0] , filtered_label1[:,1] , color = 'black', label = 1)
plt.legend()
plt.show()