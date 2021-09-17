import argparse
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats
from tqdm import tqdm
from torch.utils.data import DataLoader
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
    utils.log('dist method: {} logits_coeff_list: {} , feat_source_list: {} branch_list: {}'.format(
        args.method, args.logits_coeff_list, args.feat_source_list, args.branch_list))

    # Datasets.
    dataset = datasets.make(config['dataset'], **config['dataset_args'])
    utils.log('dataset: {} (x{}), {}'.format(
            dataset[0][0].shape, len(dataset), dataset.n_classes))
    n_way = 5
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
        
        # Read data.
        model_data = torch.load(config.get('load'))
        
        # Load model.
        model = models.load(model_data)
        # Other settings for models.
        if config.get('_parallel'):
            model = nn.DataParallel(model)
            model = convert_model(model).to('cuda')
        

    # Switch model to eval mode     
    model.eval()
    
    # Show properties of models.
    utils.log('num params: {}'.format(utils.compute_n_params(model)))


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
                
                logits_list = []
                # Consider multiple models.
                
                # Get settings for features to be extracted.
                method_i_list = args.method.split(',')         # Always use the method from arguments.
                logits_coeff_list = list(map(eval, args.logits_coeff_list.split(',')))  # Note: eval() is dangerous. Be careful when use it.
                
                branch_list = list(map(int, args.branch_list.split(',')))

                feat_source_list = args.feat_source_list.split(',') # Use feature source list only when side outputs are enabled.


                # Procedure to get logits for current model.
                method_logits_list = []
                for method_i, logits_coeff, feat_source, branch in zip(method_i_list, logits_coeff_list, feat_source_list, branch_list):
                     # Get embeddings.

                    x_query_out, x_shot_out, s_query_out, s_shot_out = model(mode='meta_test', x_shot=x_shot_origin, x_query=x_query_origin, sideout=True, branch=branch)


                    # Select embeddings.
                    if feat_source == 'final':
                        x_query_pre = x_query_out
                        x_shot_pre = x_shot_out
                    else:
                        x_query_pre = s_query_out[feat_source]
                        x_shot_pre = s_shot_out[feat_source]

                    # Pre-process embeddings.
                    x_shot_pre = x_shot_pre.mean(dim=2)  # Shape: [#tasks,#classes,#shots,...]. ProtoNet: Average embeddings.
                    
                    x_shot = x_shot_pre.view(*x_shot_pre.shape[:2], -1)
                    x_query = x_query_pre.view(*x_query_pre.shape[:2], -1)
                        
                    
                    # Process embeddings.
                    # -  Methods without the need for distance function (for compatibility).
                    if method_i == 'cos':
                        x_shot_processed = F.normalize(x_shot, dim=-1)
                        x_query_processed = F.normalize(x_query, dim=-1)
                    else:
                        raise NotImplementedError()

                    # Calculate logits with proper distance function.
                    assert x_query.dim() == x_shot.dim() == 3
                    if method_i in ['cos']:
                        logits = torch.bmm(x_query_processed, x_shot_processed.permute(0, 2, 1))
                    else:
                        raise NotImplementedError()


                    # Accumulate logits for current model.
                    method_logits_list.append(logits * logits_coeff)

                # Accumulate logits for all models.
                logits = sum(method_logits_list).view(-1, n_way)
                

                # Calculate the accuracy and loss.
                
                label = fs.make_nk_label(n_way, n_query,
                        ep_per_batch=ep_per_batch).cuda()
                loss = F.cross_entropy(logits, label)
                acc = utils.compute_acc(logits, label)

                aves['vl'].add(loss.item(), len(data))
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
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--method', default='cos') 
    parser.add_argument('--load_encoder', type=str, default='')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--log_filename', type=str, default='log.txt')
    parser.add_argument('--vscode_debug', action='store_true', default=False)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--logits_coeff_list', type=str, default='1.0')
    parser.add_argument('--sideout', action='store_true', default=False)
    parser.add_argument('--feat_source_list', type=str, default='')
    parser.add_argument('--branch_list', type=str, default='')
    args = parser.parse_args()

    # Set debug options.
    if args.vscode_debug:
        # Ref: https://vinta.ws/code/remotely-debug-a-python-app-inside-a-docker-container-in-visual-studio-code.html
        import debugpy
        print("Enabling attach starts.")
#         ptvsd.enable_attach(address=('0.0.0.0', 9310))
#         ptvsd.wait_for_attach()
        
        debugpy.listen(address=('0.0.0.0', 9310))
        debugpy.wait_for_client()
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

