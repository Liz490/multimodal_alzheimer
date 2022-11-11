import os 
import glob
import pandas as pd
import sys
from datetime import datetime
import json
import sys


f_tab_adni_tau = pd.read_csv('./data/ADNI_Tau_Amyloid_SUVR_amyloid_tau_status_dems.csv')


with open('data_set_split.json', 'r') as f:
    dict_split = json.load(f)

for mode in ['train', 'val', 'test']:


    list_ids = dict_split[mode]

    df_pet_av1451 = pd.DataFrame(columns=['ID', 'ses', 'path'])
    not_available = 0

    for id in sorted(os.listdir("/vol/chameleon/projects/adni/data_bids_processed")):
        # go over all patient ids
        if id in list_ids:
            filepath = os.path.join("/vol/chameleon/projects/adni/data_bids_processed", id)
            modalities = os.listdir(filepath)
            # we only need the pet1451 modality for now
            if 'pet-AV1451' in modalities:
                sessionspath = os.path.join(filepath, 'pet-AV1451')
                sessions = os.listdir(sessionspath)
                # go over every single session
                for session in sessions:
                    path_ses = os.path.join(sessionspath, session)
                    files = os.listdir(path_ses)
                    relevant_files = [s for s in files if "MNI_2mm" in s]
                    # sanity check that we have an MNI registered scan available
                    if len(relevant_files) == 1:
                        p = relevant_files[0]
                        p = os.path.join(path_ses, p)
                        # create a filter to retrieve the label, each sample is uniquely defined by and id a session and the modality
                        filt = ((f_tab_adni_tau['ID'] == id) & (f_tab_adni_tau['ses'] == session) & (f_tab_adni_tau['pet.modality'] == 'pet-AV1451'))
                        label = f_tab_adni_tau[filt]['DX']
                        try:
                            label = label.iloc[0]
                            new_row = {'ID': id, 'ses': session, 'path': p, 'label': label}
                            df_pet_av1451 = df_pet_av1451.append(new_row, ignore_index=True)
                        except:
                            not_available += 1

    print(f'for {mode} split there are {not_available} labels not available!')
    path_save = os.path.join(os.getcwd(), f'data/{mode}_path_data_petav1451.csv')
    df_pet_av1451.to_csv(path_save)