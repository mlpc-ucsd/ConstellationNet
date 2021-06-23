import math

import torch
import torch.nn as nn

import models
import utils
from .models import register
import torch.nn.functional as F


@register('linear-classifier')
class LinearClassifier(nn.Module):

    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)
    

@register('nn-classifier')
class NNClassifier(nn.Module):

    def __init__(self, in_dim, n_classes, metric='cos', temp=None):
        super().__init__()
        self.proto = nn.Parameter(torch.empty(n_classes, in_dim))
        nn.init.kaiming_uniform_(self.proto, a=math.sqrt(5))
        if temp is None:
            if metric == 'cos':
                temp = nn.Parameter(torch.tensor(10.))
            else:
                temp = 1.0
        self.metric = metric
        self.temp = temp

    def forward(self, x):
        return utils.compute_logits(x, self.proto, self.metric, self.temp)



@register('classifier-sideout')
class Classifier(nn.Module):
    
    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args, sideout_info=[]):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = models.make(classifier, **classifier_args)
        
        self.sideout_info = sideout_info
        self.sideout_classifiers = nn.ModuleList()
        for _, sideout_dim in self.sideout_info:
            classifier_args['in_dim'] = sideout_dim
            self.sideout_classifiers.append(models.make(classifier, **classifier_args))

    def forward(self, x):
        x, s = self.encoder(x, sideout=True)
        out_x = self.classifier(x)
        out_s = {}
        for i, (sideout_name, _) in enumerate(self.sideout_info):
            feat_s = s[sideout_name]
            if feat_s.dim() == 4: # BCHW.
                feat_s = feat_s.mean(-1).mean(-1)  # Avgpool.
            out_s[sideout_name] = self.sideout_classifiers[i](feat_s)
        return out_x, out_s

@register('classifier-sideout-class-meta')
class Classifier(nn.Module):
    
    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args, sideout_info=[],
                 method='sqr', temp=1.0, temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)

        # Standard classifier.
        classifier_args['in_dim'] = self.encoder.out_dim
        self.classifier = models.make(classifier, **classifier_args)
        self.sideout_info = sideout_info
        self.sideout_classifiers = nn.ModuleList()
        for _, sideout_dim in self.sideout_info:
            classifier_args['in_dim'] = sideout_dim
            self.sideout_classifiers.append(models.make(classifier, **classifier_args))

        # Few-shot classifier.
        self.method = method
        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward(self, mode, x=None, x_shot=None, x_query=None, branch=-1, sideout=False):
        # Standard classifier (return logits).
        def class_forward(x, branch):
            x, s = self.encoder(x, sideout=True, branch=branch)  # Enable sideout at default.
            out_x = self.classifier(x)
            out_s = {}
            for i, (sideout_name, _) in enumerate(self.sideout_info):
                feat_s = s[sideout_name]
                if feat_s.dim() == 4: # BCHW.
                    feat_s = feat_s.mean(-1).mean(-1)  # Avgpool.
                out_s[sideout_name] = self.sideout_classifiers[i](feat_s)
            return out_x, out_s

        # Few-shot classifier (return logits).
        def meta_forward(x_shot, x_query, branch):
            shot_shape = x_shot.shape[:-3]
            query_shape = x_query.shape[:-3]
            img_shape = x_shot.shape[-3:]

            x_shot = x_shot.view(-1, *img_shape)
            x_query = x_query.view(-1, *img_shape)
            x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0), sideout=False, branch=branch)  # Disable sideout at default.
            x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
            x_shot = x_shot.view(*shot_shape, -1)
            x_query = x_query.view(*query_shape, -1)

            if self.method == 'cos':
                x_shot = x_shot.mean(dim=-2)
                x_shot = F.normalize(x_shot, dim=-1)
                x_query = F.normalize(x_query, dim=-1)
                metric = 'dot'
                logits = utils.compute_logits(x_query, x_shot, metric=metric, temp=self.temp)

            elif self.method == 'sqr':
                x_shot = x_shot.mean(dim=-2)
                metric = 'sqr'
                logits = utils.compute_logits(x_query, x_shot, metric=metric, temp=self.temp / 1600.)  # FIXME: 1600 may need to be changed according to the embedding size.
            return logits

        # Few-shot classifier (for meta test).
        def meta_test_forward(x_shot, x_query, branch, sideout=False):
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
                x_tot, s_tot = self.encoder(torch.cat([x_shot, x_query], dim=0), sideout=True, branch=branch)
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
                x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0), sideout=False, branch=branch)
                # Output shot and query.
                x_shot, x_query = x_tot[:x_shot_len], x_tot[-x_query_len:]
                feat_shape = x_shot.shape[1:]
                x_shot = x_shot.view(*shot_shape, *feat_shape)    # Keep feature shape.
                x_query = x_query.view(*query_shape, *feat_shape)
                return x_query, x_shot

        # Train or evaluation.
        if self.training:
            if mode=='class':                # For standard classification: Train.
                out_x, out_s = class_forward(x, branch=1)  # Enable sideout at default.
                return out_x, out_s

            elif mode=='meta':               # For few-shot classification: Train.
                logits = meta_forward(x_shot, x_query, branch=2)
                return logits

            else:
                raise ValueError()
        else:
            if mode=='class':                # For standard classification: Validation and test.
                out_x, out_s = class_forward(x, branch=branch)
                return out_x, out_s

            elif mode=='meta':               # For few-shot classification: Validation.
                logits = meta_forward(x_shot, x_query, branch=branch)
                return logits

            elif mode=='meta_test':          # For few-shot classification: Test.
                return meta_test_forward(x_shot, x_query, branch=branch, sideout=sideout)

            else:
                raise ValueError()



    
    
