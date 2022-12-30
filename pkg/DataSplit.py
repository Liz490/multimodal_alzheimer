#!/usr/bin/env python
import pandas as pd
import json


def split_tabular(path):
    """
    Splits tabular data by ID in a reproducible way and generates a .json file.
    Args:
        - path: path to file that contains all data samples
    """
    adni = pd.read_csv(path, low_memory=False)
    adni.dropna(subset=['DX'], inplace=True)
    ids = adni['RID'].drop_duplicates()
    test = ids.sample(frac=0.1, random_state=3551)
    ids = ids.drop(test.index)
    val = ids.sample(frac=0.1, random_state=4381)
    train = ids.drop(val.index)

    split = {'train': train.tolist(),
             'val': val.tolist(),
             'test': test.tolist()}

    with open('../data_set_split.json', 'w') as f:
        json.dump(split, f)


if __name__ == "__main__":
    PATH = '/vol/chameleon/projects/adni/Adni_merged.csv'
    split_tabular(PATH)
    print('Dataset split performed!')
