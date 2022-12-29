
import pandas as pd
import json
import os

def split_tabular():
    #read file and drop rows with missing values
    print(os.getcwd())
    schaefer_regions = pd.read_csv('../../ADNI_Tau_Amyloid_SUVR_amyloid_tau_status_dems.csv')
    schaefer_regions = schaefer_regions[schaefer_regions.columns.drop(list(schaefer_regions.filter(regex='SUVR.DKT.ROI.idx.')))]
    print(list[schaefer_regions.columns])
    schaefer_regions.drop(['sex', 'pet.modality', 'modality.description',
       'image.id', 'ses', 't.diff.uwn.psych.pet.yrs', 'ADAS13', 'MMSE', 'mPACCtrailsB',
       't.diff.adas.pet.yrs', 'PHASE', 'SITEID',
       't.diff.diagnosis.pet.yrs', 'Phase', 'tau.braak56.SUVR', 'tau.temporal.meta.SUVR', 'tau.global.SUVR', 'RID',
       'PTEDUCAT', 'ADNI_MEM', 'ADNI_EF', 'ADNI_LAN', 'ADNI_VS', 'amyloid.global.SUVR', 'amyloid.global.SUVR.status', 'centiloid',
       'tau.braak1.SUVR', 'tau.braak3.SUVR', 'tau.braak4.SUVR',
       'tau.braak5.SUVR', 'tau.braak6.SUVR', 'tau.braak34.SUVR', 'apoe'], axis = 1, inplace = True)
    print(schaefer_regions.shape[0])
    print(list[schaefer_regions.columns])
    schaefer_regions = schaefer_regions.dropna()
    print(schaefer_regions.shape[0])

    with open('data_set_split.json', 'r') as f:
        dict_split = json.load(f)

    #create new split for tabular data
    print(schaefer_regions.shape)
    print(list(schaefer_regions.columns))
    schaefer_regions = schaefer_regions.drop_duplicates(subset=['ID'], keep="first")
    schaefer_regions = schaefer_regions.set_index('ID')
    ids = schaefer_regions[schaefer_regions.index.isin(dict_split['test']) == False]

    test = schaefer_regions[schaefer_regions.index.isin(dict_split['test']) == True]
    val = ids.sample(frac=0.1, random_state=4381)
    train = ids.drop(val.index)

    split = {'train': train.index.values.tolist(),
             'val': val.index.values.tolist(),
             'test': test.index.values.tolist(),}

    with open('data_split_tabular_schaefer.json', 'w') as f:
        json.dump(split, f)

if __name__ == "__main__":
    split_tabular()
    print('Dataset split performed!')