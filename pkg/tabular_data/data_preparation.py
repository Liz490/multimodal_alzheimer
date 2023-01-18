"""Prepares data for training the baseline model for tabular data

This file contains all relevant functions to retrieve specific values
from the overall Adni-table that contains all the tabular data available.
Furthermore, it provides a function to exclude/include MCI data.

"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from pkg.dataloader import MultiModalDataset
import os
from torch.utils.data import DataLoader
from pkg.dataloader import MultiModalDataset


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
    loader = DataLoader(
        set,
        batch_size=len(set),
        shuffle=True,
        num_workers=32
    )

    batch = next(iter(loader))
    return batch['tabular'], batch['label']

def encode_labels(labels):
    """Encodes labels (MCI, AD, CN) numerically
        Args:
            labels: labels to be encoded
    """
    lab_enc = LabelEncoder()
    lab_enc.fit(labels)
    encoded = lab_enc.transform(labels)
    return encoded

if __name__ == "__main__":
    PATH = '../../Adni_merged.csv'
    cols = ['EXAMDATE', 'Ventricles', 'Hippocampus', 'WholeBrain',
            'Entorhinal', 'Fusiform', 'MidTemp', 'ICV', 'AGE',
            'DX']

    MCI = False
    EXTRACT_MCI = False
    normalise = ['Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp']
    ADNI_PATH = '../../Adni_merged.csv'
    # extractMCI_val(DSPLIT_PATH, columns, normalise)
    print('CSV files for each set was created!')
