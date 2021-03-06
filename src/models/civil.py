import os
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertTokenizerFast
from transformers import logging
from wilds.common.data_loaders import get_eval_loader
from wilds.datasets.civilcomments_dataset import CivilCommentsDataset

from .datasets import CivilComments_Batched_Dataset

logging.set_verbosity_error()

MAX_TOKEN_LENGTH = 300
NUM_CLASSES = 2

def initialize_bert_transform():
    """Adapted from the Wilds library, available at: https://github.com/p-lambda/wilds"""
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    def transform(text):
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors='pt')
        x = torch.stack(
            (tokens['input_ids'],
             tokens['attention_mask']),
            dim=2)
        x = torch.squeeze(x, dim=0) # First shape dim is always 1
        return x
    return transform


class DistilBertClassifier(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        return outputs

class Model(nn.Module):
    def __init__(self, args, weights):
        super(Model, self).__init__()
        self.num_classes = NUM_CLASSES
        self.model = DistilBertClassifier.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2,
        )
        if weights is not None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    @staticmethod
    def getDataLoaders(args, device):
        dataset = CivilCommentsDataset(root_dir=os.path.join(args.data_dir, 'wilds'), download=True)
        # get all train data
        transform = initialize_bert_transform()
        train_data = dataset.get_subset('train', transform=transform)
        # separate into subsets by distribution
        train_sets = CivilComments_Batched_Dataset(train_data, batch_size=args.batch_size)
        # take subset of test and validation, making sure that only labels appeared in train
        # are included
        datasets = {}
        for split in dataset.split_dict:
            if split != 'train':
                datasets[split] = dataset.get_subset(split, transform=transform)
        # get the loaders
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} \
            if device.type == "cuda" else {}
        train_loaders = DataLoader(train_sets, batch_size=args.batch_size, shuffle=True, **kwargs)
        tv_loaders = {}
        for split, dataset in datasets.items():
            tv_loaders[split] = get_eval_loader('standard', dataset, batch_size=256)
        return train_loaders, tv_loaders

    def forward(self, x):
        return self.model(x)
