# The repository used for hdbscan package: https://github.com/scikit-learn-contrib/hdbscan
# The example used to find out how hdbscan works: https://nbviewer.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/How%20HDBSCAN%20Works.ipynb

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as data
import hdbscan
import time


digits = data.load_digits()
test_data = digits.data
print("test data shape is: ", test_data.shape)
print(test_data)

cluster = hdbscan.HDBSCAN(min_samples=100 ,metric='euclidean', prediction_data=True)

start_time = time.time()
cluster.fit(test_data)
end_time = time.time()

print("Time is: ", (end_time-start_time))
print(cluster.probabilities_.shape)
print(cluster.probabilities_)

start_time = time.time()
softclusters = hdbscan.all_points_membership_vectors(cluster)
end_time = time.time()
print("Time is: ", (end_time-start_time))
print(softclusters.shape)
print(softclusters)


#### Experiment Hard clustering
# 500 x 64, min_samples=none, time: 0.0385          500 x 64, min_samples=100, time: 0.0539
# 5000 x 64, min_samples=none, time: 2.678          5000 x 64, min_samples=100, time: 2.965
# 50000 x 64, min_samples=none, time: 288.175       50000 x 64, min_samples=100, time: 279.432
