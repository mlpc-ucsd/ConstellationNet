# The repository used for hdbscan package: https://github.com/scikit-learn-contrib/hdbscan
# The example used to find out how hdbscan works: https://nbviewer.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/How%20HDBSCAN%20Works.ipynb

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as data
import hdbscan

# Create a example dataset
moons, _ = data.make_moons(n_samples=50, noise=0.05)
blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75,2.25),(1.0,2.0)],cluster_std=0.25)
test_data = np.vstack([moons, blobs])

plt.scatter(test_data.T[0], test_data.T[1], color='b')
plt.show()

print(test_data)

cluster = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
cluster.fit(test_data)

print(cluster)

cluster.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                      edge_alpha=0.6,
                                      node_size=80,
                                      edge_linewidth=2)

plt.show()



cluster.condensed_tree_.plot(select_clusters=True)
plt.show()

print(cluster.probabilities_)