import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import register


@register('cifar-fs')
class CIFAR_FS(Dataset):

    def __init__(self, root_path, split='train', **kwargs):
        split_tag = split
        split_file = 'CIFAR_FS_{}.pickle'.format(split_tag)

        with open(os.path.join(root_path, split_file), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
        data = pack['data']
        label = pack['labels']

        data = [Image.fromarray(x) for x in data]  # FIXME: Need to check the image size.

        old_classes = np.unique(label)
        old2new_class = {old_classes[i]:i for i in range(len(old_classes))}
        new_label = [old2new_class[l] for l in label]
        
        self.data = data
        self.label = new_label
        self.n_classes = max(self.label) + 1
        
        # Augmentation settings. Note: Here we use 'aug' instead of 'augment' in original repo.
        aug = kwargs.get('aug')

        if aug == 'long':    # (similar to) Xiaolong's settings, which is referred as 'original repo'.
            assert False 
            image_size = 80  # Note: This is different from original repo, which sets image_size = 80.
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
            mean_pix = [x/255.0 for x in [129.37731888, 124.10583864, 112.47758569]]

            std_pix = [x/255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
            normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
            
            if split=='train' :
                self.transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
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
            mean_pix = [x/255.0 for x in [129.37731888, 124.10583864, 112.47758569]]

            std_pix = [x/255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
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