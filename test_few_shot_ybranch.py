import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import os 
import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler
from pathlib import Path
from os.path import join

import sys
sys.path.append('./Synchronized-BatchNorm-PyTorch')
from sync_batchnorm import convert_model

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return h


def main(config):
    utils.log('test shot: {}'.format(args.shot))
    utils.log('dist method: {} norm_coeff: {} logits_coeff_list: {} dist_func_list: {}, feat_source_list: {} branch_list: {}'.format(
        args.method, args.norm_coeff, args.logits_coeff_list, args.dist_func_list, args.feat_source_list, args.branch_list))

    # Datasets.
    dataset = datasets.make(config['dataset'], **config['dataset_args'])
    utils.log('dataset: {} (x{}), {}'.format(
            dataset[0][0].shape, len(dataset), dataset.n_classes))
    if not args.sauc:
        n_way = 5
    else:
        n_way = 2
    n_shot, n_query = args.shot, 15
    n_batch = 800 #200 # Due to the memory issue, we change the settings. (800,1) and (200,4) lead to similar results.
    ep_per_batch = 1 #4 
    batch_sampler = CategoriesSampler(
            dataset.label, n_batch, n_way, n_shot + n_query,
            ep_per_batch=ep_per_batch)
    loader = DataLoader(dataset, batch_sampler=batch_sampler,
                        num_workers=args.num_workers, pin_memory=True)

    # Support load multiple models.
    models_list = []

    # Models from meta-training (or Y-branch model).
    if config.get('load') is not None:
        model_list_id = config.get('load')
        model_list_id = model_list_id.split(',')
        for model_i in model_list_id:
            # Read data.
            model_i_data = torch.load(model_i)
            # May overwrite model name.
            if args.disable_out_feat_flatten:
                model_i_data['model_args']['encoder_args']['out_feat_flatten'] = False
            # Load model.
            model = models.load(model_i_data)
            # Other settings for models.
            if config.get('_parallel'):
                model = nn.DataParallel(model)
                model = convert_model(model).to('cuda')
            models_list.append(model)

    [model.eval() for model in models_list]

    # Show properties of models.
    for model_i in models_list:
        utils.log('num params: {}'.format(utils.compute_n_params(model_i)))

    # Some settings for feature clustering.
    if 'cluster' in args.method:
        # Load the feature clusters.
        if args.feat_clusters_filename[0] == "/": # Use the full path instead.
            feat_clusters_path = args.feat_clusters_filename
        else:
            feat_clusters_path = join(encoder_dir, args.feat_clusters_filename)
        feat_clusters = torch.load(feat_clusters_path).cuda()
        feat_clusters_normalized = F.normalize(feat_clusters, dim=-1)

        # Function to calculate the softmax prob on cosine similarity.
        def cluster_cos_softmax(features, K=100.0):
            features = F.normalize(features, dim=-1)
            cos_dist = features.matmul(feat_clusters_normalized.transpose(0, 1))
            prob = F.softmax(K*cos_dist, dim=-1)  
            return prob

        # Function to calculate the cosine similarity.
        def cluster_cos(features):
            features = F.normalize(features, dim=-1)
            cos_dist = features.matmul(feat_clusters_normalized.transpose(0, 1))
            return cos_dist

        # Function to expand the BCHW format to BCH'W'KK shape.
        def expand_input(x, kernel_size=3, stride=1, padding=1):
            padded_input = F.pad(x, [padding,padding,padding,padding])
            out = padded_input.unfold(2,size=kernel_size,step=stride).unfold(3,kernel_size,stride)
            return out

    # Some settings for linear weights.
    if 'linear' in args.method:
        # Load the feature clusters.
        linear_weights_path = join(encoder_dir, args.linear_weights_filename)
        linear_weights = torch.load(linear_weights_path)
        linear_w = linear_weights['weight'].cuda()
        linear_b = linear_weights['bias'].cuda()
        linear_w_normalized = F.normalize(linear_w, dim=-1)

        # Function to calculate the logits based on the linear layer.
        def linear_transform(features):
            logits = features.matmul(linear_w.transpose(0, 1)) + linear_b
            return logits

        # Function to calculate the cosine similarity between features and the linear weights (w only).
        def linear_cos(features):
            features = F.normalize(features, dim=-1)
            cos_dist = features.matmul(linear_w_normalized.transpose(0, 1))
            return cos_dist

    # Testing.
    aves_keys = ['vl', 'va']
    aves = {k: utils.Averager() for k in aves_keys}

    test_epochs = args.test_epochs
    np.random.seed(0)
    va_lst = []
    for epoch in range(1, test_epochs + 1):
        for data, _ in tqdm(loader, leave=False):
            x_shot_origin, x_query_origin = fs.split_shot_query(
                    data.cuda(), n_way, n_shot, n_query,
                    ep_per_batch=ep_per_batch)

            with torch.no_grad():
                # Evaluate the few-shot classification accuracy.
                if not args.sauc:
                    logits_list = []
                    # Consider multiple models.
                    for model_i in models_list:
                        # Get settings for features to be extracted.
                        method_i_list = args.method.split(',')         # Always use the method from arguments.
                        logits_coeff_list = map(eval, args.logits_coeff_list.split(','))  # Note: eval() is dangerous. Be careful when use it.
                        dist_func_list = args.dist_func_list.split(',')
                        branch_list = map(int, args.branch_list.split(','))
                        if args.sideout:
                            feat_source_list = args.feat_source_list.split(',') # Use feature source list only when side outputs are enabled.
                        else:
                            if not args.feat_source_list:
                                feat_source_list = ['final'] * len(method_i_list)  # 'final' means only use the final layer output from the network.
                            else:
                                raise ValueError()
                        
                        # Procedure to get logits for current model.
                        method_logits_list = []
                        for method_i, logits_coeff, dist_func, feat_source, branch in zip(method_i_list, logits_coeff_list, dist_func_list, feat_source_list, branch_list):
                             # Get embeddings.
                            if args.sideout:
                                x_query_out, x_shot_out, s_query_out, s_shot_out = model_i(mode='meta_test', x_shot=x_shot_origin, x_query=x_query_origin, sideout=True, branch=branch)
                            else:
                                x_query_out, x_shot_out                          = model_i(mode='meta_test', x_shot=x_shot_origin, x_query=x_query_origin, sideout=False, branch=branch)

                            # Select embeddings.
                            if feat_source == 'final':
                                x_query_pre = x_query_out
                                x_shot_pre = x_shot_out
                            else:
                                x_query_pre = s_query_out[feat_source]
                                x_shot_pre = s_shot_out[feat_source]
                            
                            # Pre-process embeddings.
                            x_shot_pre = x_shot_pre.mean(dim=2)  # Shape: [#tasks,#classes,#shots,...]. ProtoNet: Average embeddings.
                            if args.disable_out_feat_flatten:
                                # Flatten version.
                                x_shot = x_shot_pre.view(*x_shot_pre.shape[:2], -1)
                                x_query = x_query_pre.view(*x_query_pre.shape[:2], -1)
                                # Non-flatten version.
                                x_shot_noflat = x_shot_pre
                                x_query_noflat = x_query_pre
                            else:
                                # Flatten version (at default).
                                x_shot = x_shot_pre    # TODO: May also use .view(..., -1) to flatten the features for better compatibility.
                                x_query = x_query_pre

                            # Process embeddings.
                            # -  Methods without the need for distance function (for compatibility).
                            if method_i == 'cos':
                                x_shot_processed = F.normalize(x_shot, dim=-1)
                                x_query_processed = F.normalize(x_query, dim=-1)

                            elif method_i == 'sqr':
                                x_shot_processed = x_shot  # Do not do anything.
                                x_query_processed = x_query

                            elif method_i == 'between' or method_i == 'between-another':
                                x_shot_norm = torch.norm(x_shot, dim=-1, keepdim=True)
                                x_query_norm = torch.norm(x_query, dim=-1, keepdim=True)
                                x_shot_processed = x_shot / (x_shot_norm ** args.norm_coeff)
                                x_query_processed = x_query / (x_query_norm ** args.norm_coeff)

                            elif method_i == 'only-norm' or method_i == 'only-norm-another':
                                x_shot_norm = torch.norm(x_shot, dim=-1, keepdim=True)
                                x_query_norm = torch.norm(x_query, dim=-1, keepdim=True)
                                x_shot_processed = x_shot_norm
                                x_query_processed = x_query_norm

                            elif method_i == 'single-feature':
                                x_shot_processed = x_shot[..., args.single_feature_index].unsqueeze(-1)
                                x_query_processed = x_query[..., args.single_feature_index].unsqueeze(-1)
                            
                            # - Methods that need the distance function.
                            elif method_i == 'raw':
                                x_shot_processed = x_shot  # Do not do anything.
                                x_query_processed = x_query

                            elif method_i == 'avgpool':
                                x_shot_processed = x_shot_noflat.mean(-1).mean(-1)
                                x_query_processed = x_query_noflat.mean(-1).mean(-1)

                            elif method_i == 'spatial-normalized': # [#tasks, #images, C, H, W].
                                # TODO: Use F.normalize, which is better.
                                x_shot_norm = x_shot_noflat.view(*x_shot_noflat.shape[:-2], -1).norm(dim=-1).unsqueeze(-1).unsqueeze(-1)
                                x_query_norm = x_query_noflat.view(*x_query_noflat.shape[:-2], -1).norm(dim=-1).unsqueeze(-1).unsqueeze(-1)
                                eps = 1e-7
                                x_shot_processed = (x_shot_noflat / (eps + x_shot_norm)).view(*x_shot_noflat.shape[:-3], -1)
                                x_query_processed = (x_query_noflat / (eps + x_query_norm)).view(*x_query_noflat.shape[:-3], -1)

                            elif method_i == 'spatial-norm':
                                # Due to a bug in torch.norm in PyTorch 1.4, we don't use the following way to calculate norm.
                                #    x_shot_norm = x_shot_noflat.norm(dim=[-1,-2])
                                #    x_query_norm = x_query_noflat.norm(dim=[-1,-2])
                                # Ref: https://github.com/pytorch/pytorch/issues/30704
                                x_shot_norm = x_shot_noflat.view(*x_shot_noflat.shape[:-2], -1).norm(dim=-1)
                                x_query_norm = x_query_noflat.view(*x_query_noflat.shape[:-2], -1).norm(dim=-1)
                                x_shot_processed = x_shot_norm
                                x_query_processed = x_query_norm
                                
                            elif method_i == 'channelwise-normalized':
                                raise NotImplementedError()

                            elif method_i == 'channelwise-norm':
                                raise NotImplementedError()

                            elif method_i == 'cluster-cos-softmax':   # TODO: Should be correlated with feature source.
                                x_shot_processed = cluster_cos_softmax(x_shot)
                                x_query_processed = cluster_cos_softmax(x_query)

                            elif method_i == 'cluster-cos':   # TODO: Should be correlated with feature source.
                                x_shot_processed = cluster_cos(x_shot)
                                x_query_processed = cluster_cos(x_query)

                            elif method_i == 'cluster-cos-avgpool':   # TODO: Should be correlated with feature source.
                                x_shot_processed = cluster_cos(x_shot_noflat.mean(-1).mean(-1))
                                x_query_processed = cluster_cos(x_query_noflat.mean(-1).mean(-1))

                            elif method_i == 'cluster-cos-spatial1x1':   # TODO: Should be correlated with feature source.
                                B,S,C,H,W = x_shot_noflat.shape  # BSCHW.
                                _,Q,_,_,_ = x_query_noflat.shape # BQCHW.
                                x_shot_processed = cluster_cos(x_shot_noflat.permute(0,1,3,4,2).contiguous().view(-1, C))    # Shape: [BSHW,C'].
                                x_query_processed = cluster_cos(x_query_noflat.permute(0,1,3,4,2).contiguous().view(-1, C))  # Shape: [BQHW,C'].
                                x_shot_processed = x_shot_processed.view(B,S,H,W,-1).permute(0,1,4,2,3).contiguous().view(B,S,-1)   # Shape: BSHWC' -> BSC'HW -> [B,S,C'HW].
                                x_query_processed = x_query_processed.view(B,Q,H,W,-1).permute(0,1,4,2,3).contiguous().view(B,Q,-1) # Shape: BQHWC' -> BQC'HW -> [B,Q,C'HW].

                            elif method_i == 'cluster-cos-spatial1x1-avgpool':   # TODO: Should be correlated with feature source.
                                B,S,C,H,W = x_shot_noflat.shape  # BSCHW.
                                _,Q,_,_,_ = x_query_noflat.shape # BQCHW.
                                x_shot_processed = cluster_cos(x_shot_noflat.permute(0,1,3,4,2).contiguous().view(-1, C))    # Shape: [BSHW,C'].
                                x_query_processed = cluster_cos(x_query_noflat.permute(0,1,3,4,2).contiguous().view(-1, C))  # Shape: [BQHW,C'].
                                x_shot_processed = x_shot_processed.view(B,S,H,W,-1).permute(0,1,4,2,3).contiguous().mean(-1).mean(-1)   # Shape: BSHWC' -> BSC'HW -> [B,S,C'].
                                x_query_processed = x_query_processed.view(B,Q,H,W,-1).permute(0,1,4,2,3).contiguous().mean(-1).mean(-1) # Shape: BQHWC' -> BQC'HW -> [B,Q,C'].

                            elif method_i == 'cluster-cos-spatial' or method_i == 'cluster-cos-spatial-avgpool' or method_i == 'cluster-cos-spatial-softmax':   # TODO: Should be correlated with feature source.
                                B,S,C,H,W = x_shot_noflat.shape  # BSCHW.
                                _,Q,_,_,_ = x_query_noflat.shape # BQCHW.
                                x_shot_processed = x_shot_noflat.view(B*S,C,H,W)
                                x_query_processed = x_query_noflat.view(B*Q,C,H,W)
                                
                                def process(x, func=cluster_cos):
                                    x = expand_input(x, args.kernel_size, args.stride, args.padding)   # Shape: BCH'W'KK.
                                    x = x.permute(0,2,3,4,5,1).contiguous()                            # Shape: BCH'W'KK -> BH'W'KKC.
                                    B, H_prime, W_prime, _, _, _ = x.shape
                                    x = x.view(B*H_prime*W_prime, -1)   # Shape: BH'W'KKC -> [BH'W', KKC].
                                    x = func(x)                         # Shape: [BH'W', C'].
                                    x = x.view(B, H_prime, W_prime, -1).permute(0,3,1,2).contiguous()  # Shape: [BH'W',C'] -> [B,H',W',C'] -> [B,C',H',W'].
                                    return x

                                if method_i == 'cluster-cos-spatial':
                                    x_shot_processed = process(x_shot_processed).view(B,S,-1)
                                    x_query_processed = process(x_query_processed).view(B,Q,-1)
                                elif method_i == 'cluster-cos-spatial-avgpool':
                                    x_shot_processed = process(x_shot_processed).mean(-1).mean(-1).view(B,S,-1)
                                    x_query_processed = process(x_query_processed).mean(-1).mean(-1).view(B,Q,-1)
                                elif method_i == 'cluster-cos-spatial-softmax':
                                    x_shot_processed = process(x_shot_processed, func=cluster_cos_softmax).view(B,S,-1)
                                    x_query_processed = process(x_query_processed, func=cluster_cos_softmax).view(B,Q,-1)
                                else:
                                    raise ValueError()

                            elif method_i == 'linear-transform':   # TODO: Should be correlated with feature source.
                                x_shot_processed = linear_transform(x_shot)
                                x_query_processed = linear_transform(x_query)

                            elif method_i == 'linear-cos':   # TODO: Should be correlated with feature source.
                                x_shot_processed = linear_cos(x_shot)
                                x_query_processed = linear_cos(x_query)

                            else:
                                raise ValueError()
                            
                            # Calculate logits with proper distance function.
                            assert x_query.dim() == x_shot.dim() == 3
                            if method_i in ['cos', 'between-another', 'only-norm-another'] or dist_func == 'dot':
                                logits = torch.bmm(x_query_processed, x_shot_processed.permute(0, 2, 1))
                            elif method_i in ['sqr', 'between', 'only-norm', 'single-feature'] or dist_func == 'sqr':
                                logits = -(x_query_processed.unsqueeze(2) - x_shot_processed.unsqueeze(1)).pow(2).sum(dim=-1)
                            elif dist_func == 'sqr-sub':
                                logits = -(x_query_processed.unsqueeze(2) - x_shot_processed.unsqueeze(1)).pow(2) 
                                # Shape: [#tasks, #query, 1, #channels] - [#tasks, 1, #support, #channels] = [#tasks, #query, #support, #channels]
                                topk_val, _ = logits.topk(dim=-1, k=1588, largest=True)
                                topk_val, _ = topk_val.topk(dim=-1, k=250, largest=False)
                                logits = topk_val.sum(dim=-1)
                            elif dist_func == 'cos':
                                x_shot_processed = F.normalize(x_shot_processed, dim=-1)
                                x_query_processed = F.normalize(x_query_processed, dim=-1)
                                logits = torch.bmm(x_query_processed, x_shot_processed.permute(0, 2, 1))
                            else:
                                raise ValueError()

                            # Accumulate logits for current model.
                            method_logits_list.append(logits * logits_coeff)

                        # Accumulate logits for all models.
                        methods_logits = sum(method_logits_list)  # TODO: May consider product in future.
                        methods_logits = methods_logits           # Remove the temperature.
                        logits_list.append(methods_logits.view(-1, n_way))

                    # Calculate the accuracy and loss.
#                     logits = np.add.reduce(logits_list)
                    logits = sum(logits_list)
                    label = fs.make_nk_label(n_way, n_query,
                            ep_per_batch=ep_per_batch).cuda()
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)

                    aves['vl'].add(loss.item(), len(data))
                    aves['va'].add(acc, len(data))
                    va_lst.append(acc)
                else:
                    x_shot = x_shot[:, 0, :, :, :, :].contiguous()
                    shot_shape = x_shot.shape[:-3]
                    img_shape = x_shot.shape[-3:]
                    bs = shot_shape[0]
                    p = model.encoder(x_shot.view(-1, *img_shape)).reshape(
                            *shot_shape, -1).mean(dim=1, keepdim=True)
                    q = model.encoder(x_query.view(-1, *img_shape)).view(
                            bs, -1, p.shape[-1])
                    p = F.normalize(p, dim=-1)
                    q = F.normalize(q, dim=-1)
                    s = torch.bmm(q, p.transpose(2, 1)).view(bs, -1).cpu()
                    for i in range(bs):
                        k = s.shape[1] // 2
                        y_true = [1] * k + [0] * k
                        acc = roc_auc_score(y_true, s[i])
                        aves['va'].add(acc, len(data))
                        va_lst.append(acc)

        utils.log('test epoch {}: acc={:.2f} +- {:.2f} (%), loss={:.4f}'.format(
                epoch, aves['va'].item() * 100,
                mean_confidence_interval(va_lst) * 100,
                aves['vl'].item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config', default='./configs/test_few_shot.yaml')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test-epochs', type=int, default=10)
    parser.add_argument('--sauc', action='store_true')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--method', default='cos')  # Can be 'cos', 'sqr', 'between', 'between-another', 'only-norm', 'only-norm-another', 'single-feature', etc.
    parser.add_argument('--norm_coeff', type=float, default=0.5)
    parser.add_argument('--single_feature_index', type=int, default=0)
    parser.add_argument('--load_encoder', type=str, default='')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--log_filename', type=str, default='log.txt')
    parser.add_argument('--vscode_debug', action='store_true', default=False)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--logits_coeff_list', type=str, default='1.0')
    parser.add_argument('--disable_out_feat_flatten', action='store_true')
    parser.add_argument('--dist_func_list', type=str, default='none')
    parser.add_argument('--feat_clusters_filename', type=str, default='')
    parser.add_argument('--linear_weights_filename', type=str, default='')
    parser.add_argument('--sideout', action='store_true', default=False)
    parser.add_argument('--feat_source_list', type=str, default='')
    parser.add_argument('--kernel_size', default=1, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--padding', default=0, type=int)
    parser.add_argument('--branch_list', type=str, default='')
    args = parser.parse_args()

    # Set debug options.
    if args.vscode_debug:
        # Ref: https://vinta.ws/code/remotely-debug-a-python-app-inside-a-docker-container-in-visual-studio-code.html
        import ptvsd
        print("Enabling attach starts.")
        ptvsd.enable_attach(address=('0.0.0.0', 9310))
        ptvsd.wait_for_attach()
        print("Enabling attach ends.")

    # Load and overwrite configs.
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
    if args.load_encoder:  # Overwrite the path of encoder checkpoint to load.
        config['load_encoder'] = args.load_encoder
    if args.load:
        config['load'] = args.load
    
    # Other settings.
    if args.save_path: # Specify the path to save logs.
        os.makedirs(args.save_path, exist_ok=True)
        utils.set_log_path(args.save_path)
    else:
        load_path = Path(args.load)        # Modified from load_encoder to load.
        model_dir = str(load_path.parent)
        utils.set_log_path(model_dir)

    utils.set_log_filename(args.log_filename)
    utils.set_gpu(args.gpu)

    # Main function.
    main(config)

