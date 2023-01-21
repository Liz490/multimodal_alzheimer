"""Prepares data for training the baseline model for tabular data

This file contains all relevant functions to retrieve specific values
from the overall Adni-table that contains all the tabular data available.
Furthermore, it provides a function to exclude/include MCI data.

"""
import os
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pkg.utils.dataloader import MultiModalDataset


BASEPATH = os.getcwd()
TRAINPATH = os.path.join(BASEPATH, 'data/train_path_data_labels.csv')
VALPATH = os.path.join(BASEPATH, 'data/val_path_data_labels.csv')

def get_data(path, binary_classification):
    """retrieves training and validation data in suitable format
        Args:
            val_data_path: path to file containins validation data
            train_data_path: path to file containins training data
    """
    set = MultiModalDataset(path=path,
                               binary_classification=binary_classification,modalities=['tabular'])
    print(path)
    loader = DataLoader(
        set,
        batch_size=len(set),
        shuffle=True,
        num_workers=32
    )
    batch = next(iter(loader))
    return batch['tabular'], batch['label']
