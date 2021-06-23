import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register
from collections import OrderedDict

@register('simple-encoder-sideout')
class SimpleEncoder(nn.Module):

    def __init__(self, encoder, temp=1., method='cos', encoder_args={}):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.temp = temp
        self.method = method

    def forward(self, x_shot, x_query, sideout=False):
        # Get shapes and reshape shot and query.
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]
        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_shot_len = len(x_shot)
        x_query_len = len(x_query)

        if sideout:
            # Network outputs.
            x_tot, s_tot = self.encoder(torch.cat([x_shot, x_query], dim=0), sideout=True)
            # Output shot and query.
            x_shot, x_query = x_tot[:x_shot_len], x_tot[-x_query_len:]
            feat_shape = x_shot.shape[1:]
            x_shot = x_shot.view(*shot_shape, *feat_shape)    # Keep feature shape.
            x_query = x_query.view(*query_shape, *feat_shape)
            # Side output shot and query.
            s_shot  = {k:v[  :x_shot_len].view(*shot_shape,  *v.shape[1:]) for k, v in s_tot.items()}
            s_query = {k:v[-x_query_len:].view(*query_shape, *v.shape[1:]) for k, v in s_tot.items()}
            return x_query, x_shot, s_query, s_shot
        else:
            # Network outputs.
            x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
            # Output shot and query.
            x_shot, x_query = x_tot[:x_shot_len], x_tot[-x_query_len:]
            feat_shape = x_shot.shape[1:]
            x_shot = x_shot.view(*shot_shape, *feat_shape)    # Keep feature shape.
            x_query = x_query.view(*query_shape, *feat_shape)
            return x_query, x_shot


@register('simple-encoder')
class SimpleEncoder(nn.Module):

    def __init__(self, encoder, temp=1., method='cos', encoder_args={}):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.temp = temp
        self.method = method

    def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        feat_shape = x_shot.shape[1:]
        x_shot = x_shot.view(*shot_shape, *feat_shape)    # Keep feature shape.
        x_query = x_query.view(*query_shape, *feat_shape)
        
        return x_query, x_shot