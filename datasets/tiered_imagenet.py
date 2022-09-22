import os
import pickle
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os.path as osp
from .datasets import register


@register('tiered-imagenet')
class TieredImageNet(Dataset):

    def __init__(self, root_path, split='train', **kwargs):
        TRAIN_PATH = osp.join(root_path, 'tiered_imagenet/train')
        VAL_PATH = osp.join(root_path, 'tiered_imagenet/val')
        TEST_PATH = osp.join(root_path, 'tiered_imagenet/test')
        if split == 'train':
            THE_PATH = TRAIN_PATH
        elif split == 'test':
            THE_PATH = TEST_PATH
        elif split == 'val':
            THE_PATH = VAL_PATH
        else:
            raise ValueError('Wrong split.')
        data = []
        label = []
        folders = [osp.join(THE_PATH, label) for label in os.listdir(THE_PATH) if
                   os.path.isdir(osp.join(THE_PATH, label))]

        for idx in range(len(folders)):
            this_folder = folders[idx]
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)
        def convert_raw(x):
            mean = torch.tensor([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]).view(3, 1, 1).type_as(x)
            std = torch.tensor([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw
        self.data = data
        self.label = label
        self.n_classes = len(set(label))

        # Transformation
        image_size = 84
        self.transform_aug = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                    np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))]
            )

        self.transform_normal = transforms.Compose([
            transforms.Resize([92, 92]),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                    np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))])
        num_sampling = kwargs.get('num_sampling', None)
        if num_sampling is None:
            self.use_lrd = False
            augment = kwargs.get('augment',None)
            if augment is None:
                self.transform = self.transform_normal
            else:
                self.transform = self.transform_aug
        else:
            self.use_lrd = True
            self.num_sampling = num_sampling
            self.transform = self.transform_aug

        if kwargs.get('transform', None): # overwrite transform with the provided one
            self.transform = kwargs.get('transform', None)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if type(idx) == tuple:
            i = idx[0]
            if idx[1] == 1:
                trans_func = self.transform_aug
            else:
                trans_func = self.transform_normal
        else:
            i = idx
            trans_func = self.transform
        if self.use_lrd:
            path, label = self.data[i], self.label[i]
            img_list=[]
            for _ in range(self.num_sampling):
                img_list.append(trans_func(Image.open(path).convert('RGB')))
            img_list=torch.stack(img_list,dim=0)
            return img_list, label
        else:
            path, label = self.data[i], self.label[i]
            image = trans_func(Image.open(path).convert('RGB'))
            return image, label

