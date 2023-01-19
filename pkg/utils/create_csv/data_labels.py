"""
Python script that creates a .csv file containing all relevant data and labels for the multi-modal DL task.

Executing the script creates three files in the data folder that contain the inputs and labels for the 
train, validation and test split according to the ID's listed in 'data_set_split.json':
    - train_path_data_labels.csv
    - val_path_data_labels.csv
    - test_path_data_labels.csv
Each row in these files corresponds to one single modality (PET, MRI or tabular) and multiple lines can be merged for the multi-modal 
training if they belong to the same patient and if the data was acquired within a certain amount of time.
The columns of the dataframe include:
    - a unique subject ID: ID
    - a string with the date indicating when the data was acquired: ses
    - PET modality related data: path_pet1451
    - MRI modality related data: path_anat, path_anat_mask
    - tabular data: AGE, PTEDUCAT, Ventricles, Hippocampus, WholeBrain, Entorhinal, Fusiform, MidTemp, ICV
    - labels: label
Each row consists of ID, ses and label but only contains data for either one of the three modalities. All the remaining columns
hold NaN values.
"""

import os 
import glob
import pandas as pd
import sys
from datetime import datetime, timedelta
import json
import sys
import numpy as np
from torch import threshold
from typing import Tuple
#######################################################################
# TODO:
#  1) maybe use other split! So far only tabular data of subjects that also have an
#     image modality in data_bids_processed are considered
#  
#
#
#######################################################################

# define helper functions
def get_timedelta_from_string(timestring: str, 
                            format: str="ses-%Y-%m-%d",
) -> datetime:
    """
    Convert a time in string format to timedelta.

    Args:
    timestring: A string that holds date information
    format: A string that specifies the format of the timestring

    Returns:
    The converted datetime timedelta object from the string
    """
    td = datetime.strptime(timestring, format)
    return td
    
def get_rid_from_id(id_string: str) -> int:
    """
    Retrieve the patient ID as int.

    Args:
    id_string: A string where the last 4 characters consist of the ID

    Returns:
    An int obtained from the string
    """
    id_string = id_string[-4:]
    id_int = int(id_string)
    return id_int

def find_closest_timestamp(date: datetime,
                         df: pd.DataFrame,
                         col_name: str='EXAMDATE',
) -> Tuple[int, int]:
    """
    Given a date search for the row in a pd.DataFrame with the smallest temporal difference.

    Args:
    date: A datetime object for which we want to find the closest timestamp
    df: The DataFrame in which the row with the closest temporal difference is to be found
    col_name: A string that indicates the name of the column in the DataFrame that holds the date

    Returns:
    A Tuple that indicates 
        the temporal difference of the obtained row in days
        and the index of the row in the DataFrame
    """
    df_date = df.copy()
    # remove NaN values in EXAMDATE column
    df_date = df_date.dropna(subset=col_name)
    df_date[col_name] = df_date[col_name].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    # calculate the time difference as int with a new helper column 'time_diff'
    df_date['time_diff'] = (date - df_date[col_name])
    df_date['time_diff'] = df_date['time_diff'].apply(lambda x: x.days)
    df_date['time_diff'] = df_date['time_diff'].astype(int)
    # we don't care about the sign just relative time distance
    df_date['time_diff'] = df_date['time_diff'].apply(lambda x: np.abs(x))

    days, idx = df_date['time_diff'].min(), df_date['time_diff'].idxmin()
    return days, idx

def get_diag(row: pd.Series) -> str:
    """
    Convert the diagnosis used in the ADNI database to one of the three class labels used in the task.
        'DXCURREN' and 'DIAGNOSIS':
            1=CN, 2=MCI, 3=DEMENTIA
        'DXCHANGE':
            1=CN->CN
            2=MCI->MCI
            3=DEMENTIA->DEMENTIA
            4=CN->MCI
            5=MCI->DEMENTIA
            6=CN->DEMENTIA
            7=MCI->CN
            8=DEMENTIA->MCI
            9=DEMENTIA->CN
    
    Args:
    row: Pandas Series from which we want to retrieve a label

    Returns:
    A string that holds the converted label
    """
    if (row['DXCURREN']==1) or (row['DXCHANGE']==1) or (row['DXCHANGE']==7) or (row['DXCHANGE']==9) or (row['DIAGNOSIS']==1):
        return 'CN'
    elif (row['DXCURREN']==2) or (row['DXCHANGE']==2) or (row['DXCHANGE']==4) or (row['DXCHANGE']==8) or (row['DIAGNOSIS']==2):
        return 'MCI'
    elif (row['DXCURREN']==3) or (row['DXCHANGE']==3) or (row['DXCHANGE']==5) or (row['DXCHANGE']==6) or (row['DIAGNOSIS']==3):
        return 'Dementia'
    else:
        return 'not defined'


# Import csv files that contain label information
f_tab_adni_tau = pd.read_csv('./data/ADNI_Tau_Amyloid_SUVR_amyloid_tau_status_dems.csv', low_memory=False)
diagnosis_dxsum = pd.read_csv('/vol/chameleon/projects/adni/Diagnosis/DXSUM_PDXCONV_ADNIALL.csv', low_memory=False)
# Import csv file that contains data used for the tabular model
# Define the relevant features from the tabular data that we want to extract
relevant_feats_tab = ['RID', 'EXAMDATE', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV', 'AGE', 'Years_bl', 'PTEDUCAT', 'DX']
f_tab_data = pd.read_csv('/vol/chameleon/projects/adni/Adni_merged.csv', low_memory=False, usecols=relevant_feats_tab)
f_tab_data['AGE'] = f_tab_data['AGE'] + f_tab_data['Years_bl']
f_tab_data = f_tab_data.drop(columns='Years_bl')
# TO CHECK:
# print(f_tab_data.head(15))
# # # print(f_tab_data['Years_bl'].isnull().values.any())
# sys.exit()


# convert EXAMDATE column to type timedelta
f_tab_data['EXAMDATE'] = f_tab_data['EXAMDATE'].apply(lambda x: datetime.strptime(x, '%d/%m/%Y'))
print('We are dropping all lines in the tabular dataframe where at least one value is missing!!')
print(f'number of entries before: {len(f_tab_data)}')
f_tab_data = f_tab_data.dropna()
print(f'number of entries after: {len(f_tab_data)}')


# number of days that the date of a diagnosis can be away from the date where the mri t1w was taken
THRESHOLD_DAYS_MRI = 150 

# open json that holds information about the split
with open('data_set_split.json', 'r') as f:
    dict_split = json.load(f)

# crate a csv file for each split 
for mode in ['train', 'val', 'test']:
    # retrieve the list of IDs for the current split
    list_ids = dict_split[mode]
    
    # init df that contains the paths to the image data 
    data_paths = pd.DataFrame()

    # count non-available samples
    not_available_pet1451 = 0
    not_available_anat = 0

    # iterate over all IDs in data_bids_processed
    for id in sorted(os.listdir("/vol/chameleon/projects/adni/data_bids_processed")):
        # check if patient ID is in the current split
        if id in list_ids:
            idpath = os.path.join("/vol/chameleon/projects/adni/data_bids_processed", id)
            # get a list with all the modalities that are available for the patient
            modalities = os.listdir(idpath)

            # check for all three modalities
            
            ###############
            # pet-AV1451  #
            ###############
            if 'pet-AV1451' in modalities:
                sessionspath_pet1451 = os.path.join(idpath, 'pet-AV1451')
                # get a list with all pet-AV1451 sessions for the patient
                sessions_pet1451 = os.listdir(sessionspath_pet1451)
                sessions_pet1451 = [x for x in sessions_pet1451 if 'ses' in x]
                # go over every single session
                for session in sessions_pet1451:
                    path_ses = os.path.join(sessionspath_pet1451, session)
                    files_pet1451 = os.listdir(path_ses)
                    # we only want the MNI_2mm for training
                    relevant_files_pet1451 = [s for s in files_pet1451 if "MNI_2mm" in s]
                    # sanity check that we have an MNI registered scan available
                    # this is always either a list of length 1 or an empty list
                    if len(relevant_files_pet1451) == 1:
                        p = relevant_files_pet1451[0]
                        path_mni2mm = os.path.join(path_ses, p)
                        # create a filter to retrieve the label, each sample is uniquely defined by and id a session and the modality
                        filt = ((f_tab_adni_tau['ID'] == id) & (f_tab_adni_tau['ses'] == session) & (f_tab_adni_tau['pet.modality'] == 'pet-AV1451'))
                        # a pandas series that contains the label
                        label = f_tab_adni_tau[filt]['DX']
                        # sometimes we get an empty dataframe because the id is not in the table
                        # TODO check if we have a more complete table
                        try:
                            label = label.iloc[0]
                            session_timedelta = get_timedelta_from_string(session)
                            new_row = pd.Series({'ID': id, 'ses': session_timedelta, 'path_pet1451': path_mni2mm, 'label': label})
                            # add the new row to the df
                            data_paths = pd.concat([data_paths, new_row.to_frame().T], ignore_index=True)
                        except:
                            not_available_pet1451 += 1

            #######################
            #  anat MRI T1w       #
            #######################
            if 'anat' in modalities:
                sessionspath_anat = os.path.join(idpath, 'anat')
                # get a list with all anat sessions for the patient
                sessions_anat = os.listdir(sessionspath_anat)
                sessions_anat = [x for x in sessions_anat if 'ses' in x]
                # go over every single session
                for session in sessions_anat:

                    path_ses = os.path.join(sessionspath_anat, session)
                    files_anat = os.listdir(path_ses)
                    # we only want the MNI_2mm for training
                    relevant_files_anat = [s for s in files_anat if "reg_ants2_MNI_2mm" in s]

                    # additionally we need  the path to the brain mask
                    mask_path = os.path.join(path_ses, 'antsCorticalThickness/BrainExtractionMask_ants2_MNI_2mm.nii.gz')

                    if len(relevant_files_anat) == 1:
                        p = relevant_files_anat[0]
                        path_mni2mm = os.path.join(path_ses, p)

                        # OBTAIN CORRESPONDING LABEL    
                        # get a timedelta representation of the string
                        session_timedelta = get_timedelta_from_string(session)

                        #convert id string to integer
                        rid = get_rid_from_id(id)

                        # df that holds all the subject related information
                        df_subject = diagnosis_dxsum.loc[diagnosis_dxsum['RID'] == rid]
                        
                        # check if rid from data_bids_processed has an entry in the diagnosis dataframe
                        if df_subject.empty:
                            not_available_anat += 1
                        else:

                            # get the number of days and the row for the closest available diagnosis
                            days, index_subject = find_closest_timestamp(date=session_timedelta, df=df_subject)
                            
                            if days < THRESHOLD_DAYS_MRI:
                                # extract data for relevant row
                                row_subject = df_subject.loc[index_subject]
                                # extract diagnosis
                                label = get_diag(row_subject)
                                
                                new_row = pd.Series({'ID': id, 'ses': session_timedelta, 'path_anat': path_mni2mm, 'path_anat_mask': mask_path,'label': label})
                                # add the new row to the df that holds the paths
                                data_paths = pd.concat([data_paths, new_row.to_frame().T], ignore_index=True)
                            else:
                                not_available_anat += 1
    #################
    # tabular data  #
    #################      
    f_tab_data_mode = f_tab_data[f_tab_data['RID'].isin(list_ids)]    
    f_tab_data_mode = f_tab_data_mode.rename(columns = {'RID': 'ID', 'EXAMDATE': 'ses', 'DX': 'label'})
    data_paths = pd.concat([data_paths, f_tab_data_mode], ignore_index=True)
                        

    print(f'for {mode} split there are {not_available_pet1451} PET1451 and {not_available_anat} ANAT labels not available!')
    
    path_save = os.path.join(os.getcwd(), f'data/{mode}_path_data_labels.csv')
    
    data_paths.to_csv(path_save)
