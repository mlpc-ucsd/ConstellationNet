import os
import pickle
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import register


@register('tiered-imagenet')
class TieredImageNet(Dataset):

    def __init__(self, root_path, split='train', mini=False, **kwargs):
        split_tag = split
        data = np.load(os.path.join(
                root_path, '{}_images.npz'.format(split_tag)),
                allow_pickle=True)['images']
        data = data[:, :, :, ::-1]
        with open(os.path.join(
                root_path, '{}_labels.pkl'.format(split_tag)), 'rb') as f:
            label = pickle.load(f)['labels']

        print("Loaded files.")
        data = [Image.fromarray(x) for x in data]
        print("Loaded images.")
        
        min_label = min(label)
        label = [x - min_label for x in label]

        if mini:
            data_ = []
            label_ = []
            np.random.seed(0)
            c = np.random.choice(max(label) + 1, 64, replace=False).tolist()
            n = len(data)
            cnt = {x: 0 for x in c}
            ind = {x: i for i, x in enumerate(c)}
            for i in range(n):
                y = int(label[i])
                if y in c and cnt[y] < 600:
                    data_.append(data[i])
                    label_.append(ind[y])
                    cnt[y] += 1
            data = data_
            label = label_

        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1

        aug = kwargs.get('aug')

        if aug == 'long': 

            image_size = 80
            norm_params = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
            normalize = transforms.Normalize(**norm_params)
            self.default_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                normalize,  
            ])
            augment = kwargs.get('augment')
            if augment == 'resize':
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif augment == 'crop':
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.RandomCrop(image_size, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif augment is None:
                self.transform = self.default_transform

            def convert_raw(x):
                mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
                std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
                return x * std + mean
            self.convert_raw = convert_raw
        elif aug == 'lee':
            mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
            std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
            normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
            
            if split=='train' :
                self.transform = transforms.Compose([
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transform = transforms.Compose([
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    normalize
                ])

        elif aug == 'lee-test': # Kwonjoon Lee's settings: Always use the test settings.
            mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
            std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]
            normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

            self.transform = transforms.Compose([
                lambda x: np.asarray(x),
                transforms.ToTensor(),
                normalize
            ])
        else:
            raise ValueError()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]
