import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import register, make
from .dropblock import DropBlock
from .featcluster_minibatch import FeatureClusteringMinibatch
from .self_attention import Transformer


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class MergeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_feat_cluster=False,use_self_attention=False,
                 self_attention_kwargs={},
                 feat_cluster_kwargs={}):
        super().__init__() 
   
        # global settings.
        self.use_feat_cluster = use_feat_cluster
        self.use_self_attention = use_self_attention
        self.self_attention_kwargs = self_attention_kwargs
        self.feat_cluster_kwargs = feat_cluster_kwargs
        
        self.conv = conv3x3(in_channels, out_channels)
        self.feat_cluster_kwargs['channels'] = out_channels
        if self.use_self_attention:
            
            # Cell feature relation modeling layer
            self.transformer = Transformer(**self.self_attention_kwargs)
            
            # Positional encoding layer
            self.pe = make(self.self_attention_kwargs['positional_encoding'],\
                      num_pos_feats=self_attention_kwargs['embedding_size']//2)
            
         
        if self.use_feat_cluster:
            # Feature clustering layer.
            self.feat_cluster = FeatureClusteringMinibatch(**self.feat_cluster_kwargs)
            # Merging layer.            
            self.merge = nn.Conv2d(out_channels + self.feat_cluster_kwargs['num_clusters'], out_channels, kernel_size=1, stride=1, padding=0, bias=False)  
                    

    def forward(self, x, sideout=False):
        B, _, H, W = x.shape
        shape = {}
        shape['B'], shape['H'], shape['W'] = B, H, W
        sideout_dict = {}
        # Convolution.
        out_conv = self.conv(x)
        sideout_dict['conv'] = out_conv
        if self.use_feat_cluster or self.use_self_attention:
            if self.use_feat_cluster:
                # Cell feature clustering.
                out_conv_reshape = out_conv.permute(0,2,3,1).contiguous().view(B*H*W, -1)
                UV_dist = self.feat_cluster(out_conv_reshape,shape)       # Shape: [B*H*W,C'].
                feature_sideout = UV_dist.view(B,H,W,-1).permute(0,3,1,2).contiguous() # Shape: [B,C',H,W]
                
            if self.use_self_attention:
                if self.self_attention_kwargs['positional_encoding'] is not None:
                    pos = self.pe(feature_sideout)
                else:
                    pos = None
                feature_sideout = self.transformer(feature_sideout.permute(0,2,3,1).contiguous(),shape,pos)
                feature_sideout = feature_sideout.permute(0,3,1,2).contiguous() # Shape: [B,C or C',H,W].
           
            sideout_dict['sideoutfeature'] = feature_sideout
            sideout_dict['sideoutfeature.avgpool'] = feature_sideout.mean(-1).mean(-1)
            
            
            out_concat = torch.cat([out_conv, feature_sideout], dim=1)
            out = self.merge(out_concat)
            sideout_dict['merge'] = out    
             
        else:
            out = out_conv
            
        # Return if enable side output.
        if sideout:
            return out, sideout_dict
        else:
            return out

        
class Block(nn.Module):
    def __init__(self, inplanes, planes, downsample, use_feat_cluster=[False, False, False], use_self_attention=[False, False, False],self_attention_kwargs={},feat_cluster_kwargs={}, 
                  retain_last_activation=True,
                 drop_rate=0.0, drop_block=False, block_size=1):
        super().__init__()

        self.retain_last_activation = retain_last_activation
        self.relu = nn.LeakyReLU(0.1)
        self.mergeblock1 = MergeBlock(inplanes, planes, use_feat_cluster[0], use_self_attention[0],self_attention_kwargs,feat_cluster_kwargs)
        self.bn1 = norm_layer(planes)
        self.mergeblock2 = MergeBlock(planes, planes, use_feat_cluster[1], use_self_attention[1],self_attention_kwargs,feat_cluster_kwargs)
        self.bn2 = norm_layer(planes)
        self.mergeblock3 = MergeBlock(planes, planes, use_feat_cluster[2], use_self_attention[2],self_attention_kwargs,feat_cluster_kwargs)
        self.bn3 = norm_layer(planes)
        self.downsample = downsample
        self.maxpool = nn.MaxPool2d(2)
        # DropBlock settings.
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def drop_layer(self, x):
        if self.drop_rate > 0:
            if self.drop_block == True: 
                feat_size = x.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(x, gamma=gamma)
            else:
                out = F.dropout(x, p=self.drop_rate, training=self.training, inplace=True)  
        else:
            out = x
        return out

    def forward(self, x, sideout=False):
        self.num_batches_tracked += 1  # For DropBlock.

        sideout_dict = {}
        if sideout:
            out, s = self.mergeblock1(x, sideout=True)
            for layer_name, layer in s.items():
                sideout_dict["mergeblock1.{}".format(layer_name)] = layer
            out = self.bn1(out)
            sideout_dict['bn1'] = out
            out = self.relu(out)
            sideout_dict['relu1'] = out
            out, s = self.mergeblock2(out, sideout=True)
            for layer_name, layer in s.items():
                sideout_dict["mergeblock2.{}".format(layer_name)] = layer
            out = self.bn2(out)
            sideout_dict['bn2'] = out
            out = self.relu(out)
            sideout_dict['relu2'] = out
            out, s = self.mergeblock3(out, sideout=True)
            for layer_name, layer in s.items():
                sideout_dict["mergeblock3.{}".format(layer_name)] = layer
            out = self.bn3(out)
            sideout_dict['bn3'] = out
            identity = self.downsample(x)
            out += identity
            if self.retain_last_activation:
                out = self.relu(out)
                sideout_dict['relu3'] = out
            out = self.maxpool(out)
            sideout_dict['maxpool'] = out
            out = self.drop_layer(out)
            sideout_dict['drop_layer'] = out
            return out, sideout_dict
        else:
            out = self.mergeblock1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.mergeblock2(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.mergeblock3(out)
            out = self.bn3(out)
            identity = self.downsample(x)
            out += identity
            if self.retain_last_activation:
                out = self.relu(out)
            out = self.maxpool(out)
            out = self.drop_layer(out)
            return out

class ResNet12(nn.Module):

    def __init__(self, channels,
                 stem_use_feat_cluster_list=[],
                 branch1_use_feat_cluster_list=[[False, False, False], [False, False, False], [False, False, False], [False, False, False]],
                 branch2_use_feat_cluster_list=[[False, False, False], [False, False, False], [False, False, False], [False, False, False]],
                 stem_use_self_attention_list=[],
                 branch1_use_self_attention_list=[],
                 branch2_use_self_attention_list=[],
                 self_attention_kwargs= {},
                 feat_cluster_kwargs={}, feat_cluster_type='input3x3', retain_last_activation=True,
                 drop_rate=0.0, dropblock_size=5, sideout_classifier=False, y_branch_stage=0):

        super().__init__()
        self.n = len(channels)
        
        
        # Prepare for arguments
        def create_list(start_stage, end_stage, use_feat_cluster_list, use_self_attention_list):
            kwargs_list = \
                [{'in_planes':          3, 'out_planes': channels[0], 'use_feat_cluster': use_feat_cluster_list[0], 'feat_cluster_kwargs': feat_cluster_kwargs, 'retain_last_activation': True, 'drop_rate': drop_rate, 'use_self_attention': use_self_attention_list[0], 'self_attention_kwargs': self_attention_kwargs },
                {'in_planes': channels[0], 'out_planes': channels[1], 'use_feat_cluster': use_feat_cluster_list[1], 'feat_cluster_kwargs': feat_cluster_kwargs, 'retain_last_activation': True, 'drop_rate': drop_rate,  'use_self_attention': use_self_attention_list[1], 'self_attention_kwargs': self_attention_kwargs},
                {'in_planes': channels[1], 'out_planes': channels[2], 'use_feat_cluster': use_feat_cluster_list[2], 'feat_cluster_kwargs': feat_cluster_kwargs, 'retain_last_activation': True, 'drop_rate': drop_rate, 'drop_block': True, 'block_size': dropblock_size, 'use_self_attention': use_self_attention_list[2], 'self_attention_kwargs': self_attention_kwargs},
                {'in_planes': channels[2], 'out_planes': channels[3], 'use_feat_cluster': use_feat_cluster_list[3], 'feat_cluster_kwargs': feat_cluster_kwargs, 'retain_last_activation': retain_last_activation, 'drop_rate': drop_rate, 'drop_block': True, 'block_size': dropblock_size, 'use_self_attention': use_self_attention_list[3], 'self_attention_kwargs': self_attention_kwargs}]
            return nn.ModuleList([self._make_layer(**kwargs_list[i]) for i in range(start_stage, end_stage+1)])
        self.stem =    create_list(0,      y_branch_stage-1, stem_use_feat_cluster_list + branch1_use_feat_cluster_list, stem_use_self_attention_list + branch1_use_self_attention_list)  # Note: Actually only stem kwargs are used. The unused branch arguments can be either from branch1 or branch2. Here we use branch1.
        self.branch1 = create_list(y_branch_stage, self.n-1, stem_use_feat_cluster_list + branch1_use_feat_cluster_list, stem_use_self_attention_list + branch1_use_self_attention_list)  # Note: Actually only branch1 kwargs are used.
        self.branch2 = create_list(y_branch_stage, self.n-1, stem_use_feat_cluster_list + branch2_use_feat_cluster_list, stem_use_self_attention_list + branch2_use_self_attention_list)  # Note: Actually only branch2 kwargs are used.

        
        self.out_dim = channels[3]
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, out_planes, use_feat_cluster, use_self_attention, self_attention_kwargs, feat_cluster_kwargs, retain_last_activation, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = nn.Sequential(
            conv1x1(in_planes, out_planes),
            norm_layer(out_planes),
        )
        block = Block(in_planes, out_planes, downsample, use_feat_cluster, use_self_attention, self_attention_kwargs, feat_cluster_kwargs,  retain_last_activation,
                      drop_rate, drop_block, block_size)
        return block

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
        
        # Feature flatten
        out = x.view(x.shape[0], -1)

        # Return if enable side output.
        if sideout:
            return out, sideout_dict
        else:
            return out

@register('resnet12-featcluster-minibatch-sideout-classifier-ybranch-param-reduced')
def resnet12(**kwargs):
    return ResNet12([64, 128, 256, 290], **kwargs)
        
        
@register('resnet12-featcluster-minibatch-sideout-classifier-ybranch')
def resnet12(**kwargs):
    return ResNet12([64, 128, 256, 512], **kwargs)