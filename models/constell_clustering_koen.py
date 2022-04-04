import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import register
import time

import hdbscan

@register('constell-clustering')
class FeatureClusteringMinibatch(nn.Module):

    def __init__(self, K=1.0, num_clusters=2,
                 fix_init=False, channels=64,
                 V_count_init=1.0, **kwargs):
        super(FeatureClusteringMinibatch, self).__init__()

        self.K = K
        self.num_clusters = num_clusters
        self.fix_init = fix_init
        self.channels = channels

        # if self.fix_init:
        #     # Use NumPy random generator, which makes sure fix_init the same between train and test.
        #     self.r = np.random.RandomState(1)
        #     V_init = torch.tensor(self.r.randn(self.num_clusters, self.channels)).float()
        # else:
        #     V_init = torch.randn(self.num_clusters, self.channels)  # Shape: [#clusters, C].
        #
        # V_init = F.normalize(V_init, dim=-1)
        #
        # self.register_buffer('V_buffer', V_init)  # Register the initialization as a buffer.
        # self.register_buffer('V_count_buffer', V_count_init * torch.ones(self.num_clusters,
        #                                                                  dtype=torch.float64))  # Cluster counter in Sculley method. Note that here is a "soft" counter.

        self.clusterer = hdbscan.HDBSCAN(min_samples=100, metric='euclidean', prediction_data=True)
        # Debugging https://github.com/scikit-learn-contrib/hdbscan/issues/100
        # Arguments for HDBSCAN() https://hdbscan.readthedocs.io/en/latest/api.html

    # def compute_dist(self, U, V):
    #     # Calculate UV distance.
    #
    #     # Normalize input.
    #
    #     V = F.normalize(V, dim=-1)  # Shape: [#clusters, C].
    #     # Calculate distance.
    #     UV_dist = U.mm(V.transpose(0, 1))  # Shape: [N, #clusters].
    #     return UV_dist

    def forward(self, x, shape={}):

        # # Load buffers.
        # V = self.V_buffer.detach().clone()
        #
        # # Note: The clone is a must! Or the tensor will get updated.
        # V_count = self.V_count_buffer.detach().clone()
        #

        # Check input.
        assert len(x.shape) == 2
        U = x  # Shape: [N, C].
        U = U.cpu().detach().numpy()  # change to numpy array

        print('Input!!!!!!!!!!!!!!!!!!!')
        print('Input shape:', U.shape)
        print(U)

        # N,C = U.shape
        # U = F.normalize(U, dim=-1)             # Shape: [N, C].
        # # Calculate distance map
        # UV_dist = self.compute_dist(U,V)
        # print("Tensor UV_dist", UV_dist)
        # UV_dist = UV_dist.cpu().detach().numpy() # change to numpy array
        # print("Numpy UV_dist", UV_dist)

        # Output
        start_time = time.time()
        print("Clustering now")
        self.clusterer.fit(U)
        print("Fitted the data")
        output = hdbscan.all_points_membership_vectors(self.clusterer)
        end_time = time.time()
        print('Output!!!!!!!!!!!!!!!!!!!')
        print("Time it took: ", (end_time-start_time))
        print('Test shape UV_dist:', output.shape)
        print(output)

        return output
        # Return UV distance.
        # UV_dist shape: [N, #clusters].