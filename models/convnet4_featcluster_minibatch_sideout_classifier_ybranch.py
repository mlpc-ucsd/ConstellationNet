import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import register,make



@register('convnet4-basic-block')
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_feat_cluster=False, use_self_attention=False, self_attention_kwargs={}, feat_cluster_kwargs={}):
        super(ConvBlock, self).__init__()
        
        # Save settings.
        self.use_feat_cluster = use_feat_cluster
        self.use_self_attention = use_self_attention
        self.self_attention_kwargs = self_attention_kwargs
        self.feat_cluster_kwargs = feat_cluster_kwargs
        self.feat_cluster_kwargs['channels'] = out_channels 
        
        # Conv layer.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if self.use_self_attention:
            
            # Cell feature relation modeling layer
            self.transformer = make('constell-attention',**self.self_attention_kwargs)
            
            # Positional encoding layer
            self.pe = make(self.self_attention_kwargs['positional_encoding'],\
                      num_pos_feats=self_attention_kwargs['embedding_size']//2)
   
        if self.use_feat_cluster:      
            
            # Feature clustering layer.
            self.feat_cluster = make('constell-clustering',**self.feat_cluster_kwargs)
            
            # Merging layer.
            self.merge = nn.Conv2d(out_channels + self.feat_cluster_kwargs['num_clusters'], out_channels, kernel_size=1, stride=1, padding=0, bias=False)  
            
        
        # Other layers
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


    def forward(self, x, sideout=False):
        
        B, _, H, W = x.shape
        shape = {}
        shape['B'], shape['H'], shape['W'] = B, H, W
        sideout_dict = {}
        out_conv = self.conv(x)
        if self.use_feat_cluster or self.use_self_attention:
            feature_sideout = out_conv # Shape: [B,C,H,W].
            
            if self.use_feat_cluster:        
                # Cell feature clustering.
                out_conv_reshape = out_conv.permute(0,2,3,1).contiguous().view(B*H*W, -1) 
                UV_dist = self.feat_cluster(out_conv_reshape, shape)       # Shape: [B*H*W,C'].            
                feature_sideout = UV_dist.view(B,H,W,-1).permute(0,3,1,2).contiguous() # Shape: [B,C',H,W]     
            
            if self.use_self_attention:
                # Cell relation modeling.
                if self.self_attention_kwargs['positional_encoding'] is not None:
                    pos = self.pe(feature_sideout)
                else:
                    pos = None         
                feature_sideout = self.transformer(feature_sideout.permute(0,2,3,1).\
                                       contiguous(),shape,pos)# Shape: [B,H,W,C]
                feature_sideout = feature_sideout.view(B,H,W,-1).permute(0,3,1,2).\
                                       contiguous() # Shape: [B,C or C',H,W].  
                
            out_concat = torch.cat([out_conv, feature_sideout], dim=1)
            out = self.merge(out_concat)
        else: 
            out = out_conv
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        if sideout:            
            return out, sideout_dict
        else:
            return out



@register('convnet4-featcluster-minibatch-sideout-classifier-ybranch')
class ConvNet4FeatCluster(nn.Module):

    def __init__(self, x_dim=3, h_dim=64, z_dim=64,  
                 stem_use_feat_cluster_list=[],
                 branch1_use_feat_cluster_list=[False, False, False, False],
                 branch2_use_feat_cluster_list=[False, False, False, False],
                 stem_use_self_attention_list=[],
                 branch1_use_self_attention_list =[False, False, False, False],
                 branch2_use_self_attention_list=[False, False, False, False],
                 self_attention_kwargs={},
                 feat_cluster_kwargs={}, y_branch_stage=0):

        super().__init__()
        channels = [h_dim, h_dim, z_dim, z_dim]# 64, 64, 64, 64
        self.n = len(channels)

        
        # Prepare for arguments
        def create_list(start_stage, end_stage, use_feat_cluster_list,use_self_attention_list):
            num_blocks = end_stage - start_stage + 1
            return nn.ModuleList(make('convnet4-basic-block',
                              in_channels = in_channels,
                              out_channels = out_channels,
                              use_feat_cluster = use_feat_cluster,
                              use_self_attention = use_self_attention,
                              self_attention_kwargs = self_attention_kwargs,
                              feat_cluster_kwargs = feat_cluster_kwargs)
                              for in_channels,out_channels,use_feat_cluster,use_self_attention,
                                self_attention_kwargs,feat_cluster_kwargs in 
                                zip(([3]+channels[:-1])[start_stage:end_stage+1],
                                   channels[start_stage:end_stage+1],
                                   use_feat_cluster_list[start_stage:end_stage+1],
                                   use_self_attention_list[start_stage:end_stage+1],
                                   [self_attention_kwargs]*num_blocks,
                                   [feat_cluster_kwargs]*num_blocks))                     
                                 
                                 

            
        self.stem =    create_list(0,      y_branch_stage-1, stem_use_feat_cluster_list + branch1_use_feat_cluster_list, stem_use_self_attention_list+branch1_use_self_attention_list)  # Note: Actually only stem kwargs are used. The unused branch arguments can be either from branch1 or branch2. Here we use branch1.
        self.branch1 = create_list(y_branch_stage, self.n-1, stem_use_feat_cluster_list + branch1_use_feat_cluster_list,stem_use_self_attention_list+branch1_use_self_attention_list)  # Note: Actually only branch1 kwargs are used.
        self.branch2 = create_list(y_branch_stage, self.n-1, stem_use_feat_cluster_list + branch2_use_feat_cluster_list,stem_use_self_attention_list+branch2_use_self_attention_list)  # Note: Actually only branch2 kwargs are used.

        
        self.out_dim = channels[3]
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x, sideout=False, branch=-1):
        # Sideout helper function.
        def sideout_func(x, sideout, attr_name):
            blocks = getattr(self, attr_name)
            if sideout:
                sideout_dict = {}
                for i in range(len(blocks)):
                    x, s = blocks[i](x, sideout=True)
                    for layer_name, layer in s.items():
                        sideout_dict["{}.{}.{}".format(attr_name, i, layer_name)] = layer
                return x, sideout_dict
            else:
                for i in range(len(blocks)):
                    x = blocks[i](x)
                return x

        if branch == 1:
            branch_attr_name = "branch1"
        elif branch == 2:
            branch_attr_name = "branch2"
        else:
            raise ValueError()

        if sideout:
            sideout_dict = {}
            x, s_stem = sideout_func(x, sideout=True, attr_name="stem")
            x, s_branch = sideout_func(x, sideout=True, attr_name=branch_attr_name)
            sideout_dict.update(s_stem)
            sideout_dict.update(s_branch)
            sideout_dict['before_avgpool'] = x
        else:
            x = sideout_func(x, sideout=False, attr_name="stem")
            x = sideout_func(x, sideout=False, attr_name=branch_attr_name)

        # Feature average pooling 
        x = x.mean(-1).mean(-1)
        
        # Return if enable side output.
        if sideout:
            return x, sideout_dict
        else:
            return x



@register('convnet4-featcluster-minibatch-sideout-classifier-ybranch-param-reduced')
def conv4_ybranch(**kwargs):
    return ConvNet4FeatCluster(x_dim=3, h_dim=64, z_dim=42,**kwargs)
