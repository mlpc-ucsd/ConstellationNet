import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import register

@register('mini-imagenet')
class MiniImageNet(Dataset):

    def __init__(self, root_path, split='train',**kwargs):
        split_tag = split
        if split == 'train':
            split_tag = 'train_phase_train'
        split_file = 'miniImageNet_category_split_{}.pickle'.format(split_tag)
        with open(os.path.join(root_path, split_file), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
        data = pack['data']
        label = pack['labels']
        min_label = min(label)
        
        data = [Image.fromarray(x) for x in data]  # FIXME: Need to check the image size.
        
        min_label = min(label)
        label = [x - min_label for x in label]
        
        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1
        
        # Augmentation settings. Note: Here we use 'aug' instead of 'augment' in original repo.
        aug = kwargs.get('aug')

        if aug == 'long':    # (similar to) Xiaolong's settings, which is referred as 'original repo'.
            image_size = 84  # Note: This is different from original repo, which sets image_size = 80.
            norm_params = {'mean': [0.485, 0.456, 0.406],
                           'std': [0.229, 0.224, 0.225]}
            normalize = transforms.Normalize(**norm_params)

            self.default_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                normalize,  
            ])

            if split=='train':
                self.transform = transforms.Compose([    # Note: This is equivalent to augment == 'resize' in original repo. The setting with augment == 'crop' is removed here.
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                self.transform = self.default_transform  # Note: In original repo, it is used when augment settings is None.

            def convert_raw(x):
                mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
                std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
                return x * std + mean
            self.convert_raw = convert_raw
            
        elif aug=='lee':  # Kwonjoon Lee's settings.
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