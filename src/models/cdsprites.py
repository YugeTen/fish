import os
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .datasets import CDsprites_Dataset, DspritesDataset

# Constants
CDspritesDatasize = torch.Size([3, 64, 64])
CDspritesChans = CDspritesDatasize[0]
NUM_CLASSES = 2

class Model(nn.Module):
    """ Classify dsprites images. """
    def __init__(self, args, weights=None):
        super(Model, self).__init__()
        self.enc = nn.Sequential(
            # input size: 1 x 64 x 64
            nn.Conv2d(CDspritesChans, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 4, 2, 0),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, NUM_CLASSES)
        )
        # c1, c2 size: latent_dim x 1 x 1
        if weights is not None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    def forward(self, x, return_feature=False):
        pre_shape = x.size()[:-3]
        e = self.enc(x.view(-1, *CDspritesDatasize))
        e = e.view(e.shape[0], -1)
        if return_feature:
            return e
        out = self.fc(e).view(*pre_shape, -1)
        return out

    @staticmethod
    def getDataLoaders(args, device):
        num_domains, batch_size = args.num_domains, args.batch_size
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}

        D = DspritesDataset(os.path.join(args.data_dir, 'cdsprites'), split=False)
        train_dataset = CDsprites_Dataset(D, args.data_dir, batch_size, num_domains, 'train')
        val_dataset = CDsprites_Dataset(D, args.data_dir, batch_size, num_domains, 'val')
        test_dataset = CDsprites_Dataset(D, args.data_dir, batch_size, num_domains, 'test')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        return train_loader, {'val': val_loader, 'test':test_loader}

