import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import register

@register('constell-clustering')
class FeatureClusteringMinibatch(nn.Module):

    def __init__(self,K=1.0,num_clusters=2, 
            fix_init=False, channels=64,  
            V_count_init=1.0,**kwargs):
        super(FeatureClusteringMinibatch, self).__init__()
      
        self.K = K
        self.num_clusters = num_clusters    
        self.fix_init = fix_init
        self.channels = channels
            
        if self.fix_init:
            # Use NumPy random generator, which makes sure fix_init the same between train and test.
            self.r = np.random.RandomState(1)
            V_init = torch.tensor(self.r.randn(self.num_clusters, self.channels)).float()
        else:
            V_init = torch.randn(self.num_clusters, self.channels)  # Shape: [#clusters, C].

        
        V_init = F.normalize(V_init, dim=-1)

        self.register_buffer('V_buffer', V_init)  # Register the initialization as a buffer.
        self.register_buffer('V_count_buffer', V_count_init * torch.ones(self.num_clusters, dtype=torch.float64))  # Cluster counter in Sculley method. Note that here is a "soft" counter.

            
            
    def compute_dist(self,U, V):
        # Calculate UV distance.
        
        # Normalize input.
        
        V = F.normalize(V, dim=-1)             # Shape: [#clusters, C].
        # Calculate distance.
        UV_dist = U.mm(V.transpose(0, 1))        # Shape: [N, #clusters]. 
        return UV_dist
        
    def mik_compute(self,UV_dist):
        # Calculate m_ik.
        Coeff = F.softmax(self.K*UV_dist, dim=-1) # Shape: [N, #clusters].
        
        return Coeff    
        
        
    def forward(self, x,shape={}):
                
        # Load buffers.
        V    = self.V_buffer.detach().clone()  
        
        # Note: The clone is a must! Or the tensor will get updated.
        V_count = self.V_count_buffer.detach().clone()
      
        # Check input.
        assert len(x.shape) == 2
        U = x                         # Shape: [N, C].

        print('Input!!!!!!!!!!!!!!!!!!!')
        print('Input shape:', U.shape)
        print(U)

        N,C = U.shape       
        U = F.normalize(U, dim=-1)             # Shape: [N, C].
        # Calculate distance map
        UV_dist = self.compute_dist(U,V)
        
        if self.training:
            # Calculate soft assignment map
            Coeff = self.mik_compute(UV_dist) # Shape: [N, #clusters].
            
            
            # Calculate v_k', sum_i(m_ik*u_i).
            cur_V = Coeff.transpose(0,1).mm(U)/Coeff.sum(0).view(-1,1) # Shape: [#clusters, C].

            # Gradually change cluster center.
            delta_count = Coeff.sum(0).double()
            V_count += delta_count
            alpha_vec = (delta_count / V_count).float().view(-1, 1)  # Shape: [#clusters, 1].
            V = (1-alpha_vec)*V + alpha_vec*cur_V
            
            # Update V and counter
            self.V_buffer.copy_(V.detach().clone()) 
            self.V_count_buffer.copy_(V_count.detach().clone())     

        print('Output!!!!!!!!!!!!!!!!!!!')
        print('Test shape UV_dist:', UV_dist.shape)
        print(UV_dist)

        return UV_dist  
        # Return UV distance.
        # UV_dist shape: [N, #clusters].