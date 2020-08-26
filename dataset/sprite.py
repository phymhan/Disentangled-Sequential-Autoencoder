import torch
import os
import glob
import numpy as np


class Sprites(torch.utils.data.Dataset):
    def __init__(self, data_root, dataset_size):
        filelist = glob.glob(os.path.join(data_root, '*.sprite'))
        self.dataset_size = min(dataset_size, len(filelist))
        self.filelist = np.random.choice(filelist, self.dataset_size)
        
    def __getitem__(self, index):
        dataitem = torch.load(self.filelist[index])
        body, shirt, pant, hair, action, sprite = dataitem.values()
        data = sprite
        return body, shirt, pant, hair, action, sprite, data

    def __len__(self):
        return self.dataset_size
