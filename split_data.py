#!/usr/bin/env python

import pandas as pd
import json


def split_tabular(path):
    """
    Split tabular data by ID in a reproducable way and generate a .json file.
    """
    df_adni_tau = pd.read_csv(path, low_memory=False)

    ids = df_adni_tau['ID'].drop_duplicates()
    test = ids.sample(frac=0.1, random_state=3551)
    ids = ids.drop(test.index)
    val = ids.sample(frac=0.1, random_state=4381)
    train = ids.drop(val.index)

    split = {'train': train.tolist(),
             'val': val.tolist(),
             'test': test.tolist()}

    with open('data_set_split.json', 'w') as f:
        json.dump(split, f)


if __name__ == "__main__":
    F_TAB_ADNI_TAU = './data/ADNI_Tau_Amyloid_SUVR_amyloid_tau_status_dems.csv'
    split_tabular(F_TAB_ADNI_TAU)
