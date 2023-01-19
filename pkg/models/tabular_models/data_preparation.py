"""Prepares data for training the baseline model for tabular data

This file contains all relevant functions to retrieve specific values
from the overall Adni-table that contains all the tabular data available.
Furthermore, it provides a function to exclude/include MCI data.

"""

import numpy as np
from sklearn.preprocessing import LabelEncoder

from pkg.utils.dataloader import MultiModalDataset


def get_data(data_path, binary_classification):
    """retrieves training and validation data in suitable format
        Args:
            val_data_path: path to file containins validation data
            train_data_path: path to file containins training data
    """
    set_tabular = MultiModalDataset(path=data_path, binary_classification=binary_classification,
                                         modalities=['tabular'])
    data = set_tabular.df_tab.to_numpy()

    # Omit index, subject-ID, examdate and label and column names
    samples = np.delete(data, [0, 1, 2, 3, 4, 5, 6], 1)
    samples = np.delete(samples, 0, axis=0)
    # Retrieve label
    labels = data[:, 6]
    labels = np.delete(labels, 0, axis=0)

    labels = encode_labels(labels)

    return samples, labels

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
