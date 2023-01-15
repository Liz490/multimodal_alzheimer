"""Prepares data for training the baseline model for tabular data

This file contains all relevant functions to retrieve specific values
from the overall Adni-table that contains all the tabular data available.
Furthermore, it provides a function to exclude/include MCI data.

"""

import json
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from pkg.dataloader import MultiModalDataset


def split_tabular(columns, path):
    """Creates data split that is in line with the data split for PET data,
     i.e. does not contain Test samples.
             Args:
                 columns: relevant columns from overall adni table
                 path: path to adni tabular data
    """
    # read file and drop rows with missing values
    print(os.getcwd())
    adni_merged = pd.read_csv(path)
    print(f'Total number of samples {adni_merged.shape[0]}')
    adni_merged = adni_merged.dropna(subset=columns)
    print(f'Number of samples with all values available: {adni_merged.shape[0]}')

    with open('../data_set_split.json', 'r', encoding="utf-8") as file:
        dict_split = json.load(file)

    adni_merged = adni_merged.drop_duplicates(subset='RID', keep="first")
    adni_merged = adni_merged.set_index('RID')
    ids = adni_merged[adni_merged.index.isin(not dict_split['test'])]

    test = adni_merged[adni_merged.index.isin(dict_split['test'])]
    val = ids.sample(frac=0.1, random_state=4381)
    train = ids.drop(val.index)

    split = {'train': train.index.values.tolist(),
             'val': val.index.values.tolist(),
             'test': test.index.values.tolist(), }

    with open('data_split.json', 'w', encoding="utf-8") as file:
        json.dump(split, file)


def write_tables(dsplit_path, adni_path, columns, mci=True, extract_mci=False, normalise=None):
    """Creates csv files for all data splits that contain relevant information
             Args:
                 dsplit_path: path to file that contains data split
                 adni_path: path to original adni table
                 columns: relevant columns from overall adni table
                 mci: If true mci labels are included
                 extract_mci: If true, only labels for mci samples are included
                 normalise: values that need to be indicated relative to intercranial volume
    """
    with open(dsplit_path, 'r', encoding="utf-8") as file:
        dict_split = json.load(file)

    adni_merged = pd.read_csv(adni_path).dropna(subset=columns)

    columns.insert(0, 'ID')
    for mode in ['train', 'val', 'test']:
        list_ids = dict_split[mode]
        df_tabular = create_csv(columns, adni_merged, list_ids, mci, extract_mci, normalise)
        path_save = os.path.join(os.getcwd(), f'./{mode}_tabular_data.csv')
        df_tabular.to_csv(path_save)


def create_csv(columns, adni_merged, list_ids, mci, extract_mci, normalise):
    """Extracts relevant information for all indicated subjects

            Args:
                columns: values that should be retrieved from overall adni table
                adni_merged: adni table from which values are extracted
                list_ids: subjects for which data should be retrieved
                mci: If true mci labels are included
                extract_mci: If true, only labels for mci samples are included
                normalise: values that need to be indicated relative to intercranial volume

            Returns:
                Pandas Dataframe containing the indicated information for the passed subjects
    """
    df_tabular = pd.DataFrame(columns=columns)

    for subject in list_ids:
        rows_df = adni_merged.loc[adni_merged['RID'] == subject]
        # sort rows by examination date
        rows_df['EXAMDATE'] = pd.to_datetime(rows_df['EXAMDATE'], format='%d/%m/%Y')
        rows_df.sort_values('EXAMDATE', inplace=True)
        i = 0

        while i < rows_df.shape[0] - 1:
            label = rows_df['DX'].iloc[i]
            if mci or label != 'MCI' or extract_mci:
                current_ses = rows_df['Month_bl'].iloc[i]
                next_ses = rows_df['Month_bl'].iloc[i + 1]

                if not extract_mci or label == 'MCI':
                    new_row = create_row(normalise, columns, rows_df.iloc[i], subject)
                    df_tabular = df_tabular.append(new_row, ignore_index=True)
                    # if next session is within 6 months, it is considered 1 examination -> skip results from next session
                if (next_ses - current_ses) < 6:
                    i += 1
            i += 1

        if i < rows_df.shape[0] and (rows_df['DX'].iloc[i] != 'MCI' or mci) and (
                not extract_mci or rows_df['DX'].iloc[i] == 'MCI'):
            new_row = create_row(normalise, columns, rows_df.iloc[i], subject)
            df_tabular = df_tabular.append(new_row, ignore_index=True)

    return df_tabular


def create_row(normalise, columns, rows_df, subject):
    """Extracts relevant information from overall adni table for a specific subject and sample

        Args:
            rows_df: relevant row from overall adni table from which values are extracted
            columns: values that should be retrieved from overall adni table
            normalise: values that need to be indicated relative to intercranial volume
            subject: id of considered subject

        Returns:
            A dic with all relevant information extracted from the overall adni table sorted according to columns of final pandas dataframe
    """
    new_row = {value: rows_df[value] for value in columns if value not in normalise and value not in ['AGE', 'ID']}
    new_row['AGE'] = rows_df['AGE'] + rows_df['Month_bl'] * (1 / 12)
    new_row['ID'] = subject

    icv = rows_df['ICV']
    normalised_row = {value: rows_df[value] / icv for value in normalise}
    final_row = {**new_row, **normalised_row}
    return final_row


def extract_mci_val(path, columns, normalise):
    """Extracts all indicated values for data samples with an MCI label.

    Args:
        path: path to file that contaisn datasplit for PET data
        columns: values that should be retrieved from overall adni table
        normalise: values that need to be indicated relative to intercranial volume
    """
    with open(path, 'r', encoding="utf-8") as file:
        dict_split = json.load(file)

    adni_merged = pd.read_csv('../../Adni_merged.csv').dropna(subset=columns)
    df_tabular = create_csv(columns, adni_merged, dict_split['val'], True, True, normalise)
    path_save = os.path.join(os.getcwd(), './val_MCI_extracted.csv')
    df_tabular.to_csv(path_save)

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
    DSPLIT_PATH = 'data_split.json'
    ADNI_PATH = '../../Adni_merged.csv'
    write_tables(DSPLIT_PATH, ADNI_PATH, cols, MCI, EXTRACT_MCI, normalise)
    # extractMCI_val(DSPLIT_PATH, columns, normalise)
    print('CSV files for each set was created!')
