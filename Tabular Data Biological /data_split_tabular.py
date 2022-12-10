
import pandas as pd
import json
import os

def split_tabular():
    #read file and drop rows with missing values
    print(os.getcwd())
    adni_merged = pd.read_csv('../../Adni_merged.csv')
    print(adni_merged.shape[0])
    adni_merged = adni_merged.dropna(subset = ['EXAMDATE', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV', 'AGE', 'DX'])
    print(adni_merged.shape[0])

    with open('../data_set_split.json', 'r') as f:
        dict_split = json.load(f)

    #create new split for tabular data
    print(adni_merged.shape)

    adni_merged = adni_merged.drop_duplicates(subset='RID', keep="first")
    adni_merged = adni_merged.set_index('RID')
    ids = adni_merged[adni_merged.index.isin(dict_split['test']) == False]

    test = adni_merged[adni_merged.index.isin(dict_split['test']) == True]
    val = ids.sample(frac=0.1, random_state=4381)
    train = ids.drop(val.index)

    split = {'train': train.index.values.tolist(),
             'val': val.index.values.tolist(),
             'test': test.index.values.tolist(),}

    with open('../data_split_tabular_bio.json', 'w') as f:
        json.dump(split, f)


if __name__ == "__main__":
    split_tabular()

    print('Dataset split performed!')