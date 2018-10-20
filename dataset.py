from torch.utils.data import Dataset
import os
import numpy as np
from skimage import io
import torch


class VerificationCodeDataset(Dataset):

    def __init__(self, data_dir, nvocab=10, transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_list = os.listdir(data_dir)
        self.nvocab = nvocab

    def __len__(self,):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.image_list[idx])
        img = io.imread(img_name)
        img = img[:32, :112, :3]
        #img = np.transpose(img, [2, 0, 1])
        labels = self.image_list[idx].split('.')[0]
        labels = label_transform(labels)

        if self.transform:
            img = self.transform(img)
        return img, labels


def label_transform(labels):
    lb = torch.zeros(len(labels)).long()
    for i in range(len(labels)):
        lb[i] = int(labels[i])
    return lb
