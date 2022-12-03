import os 
import glob
import pandas as pd
import sys
from datetime import datetime, timedelta
import json
import sys
import numpy as np
from torch import threshold
#######################################################################
# TODO:
#  1) choose adequate THRESHOLD_DAYS for mri modalities
#  2) decide which date to use (USERDATE, USERDATE2, EXAMDATE)
#  3) verify whether mri path is correct
#  4) include path to brain masks
#
#
#######################################################################

THRESHOLD_DAYS = 150 #6 month

# define helper functions

def get_timedelta_from_string(timestring, format="ses-%Y-%m-%d"):
    td = datetime.strptime(timestring, format)
    return td
    

def get_rid_from_id(id_string):
    id_string = id_string[-4:]
    id_int = int(id_string)
    return id_int


def find_closest_timestamp(date, df):
    # USERDATE: Date record created
    # USERDATE2: Date record last updated
    # EXAMDATE: Examination Date
    df_date = df.copy()

    # remove NaN values in EXAMDATE column
    df_date = df_date.dropna(subset='EXAMDATE')

    df_date['EXAMDATE'] = df_date['EXAMDATE'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    # calculate the time difference as int
    df_date['time_diff'] = (date - df_date['EXAMDATE'])
    df_date['time_diff'] = df_date['time_diff'].apply(lambda x: x.days)
    df_date['time_diff'] = df_date['time_diff'].astype(int)
    # we don't care about the sign just relative time distance
    df_date['time_diff'] = df_date['time_diff'].apply(lambda x: np.abs(x))
    
    days, idx = df_date['time_diff'].min(), df_date['time_diff'].idxmin()
    return days, idx


def get_diag(row):
    if (row['DXCURREN']==1) or (row['DXCHANGE']==1) or (row['DXCHANGE']==7) or (row['DXCHANGE']==9) or (row['DIAGNOSIS']==1):
        return 'CN'
    elif (row['DXCURREN']==2) or (row['DXCHANGE']==2) or (row['DXCHANGE']==4) or (row['DXCHANGE']==8) or (row['DIAGNOSIS']==2):
        return 'MCI'
    elif (row['DXCURREN']==3) or (row['DXCHANGE']==3) or (row['DXCHANGE']==5) or (row['DXCHANGE']==6) or (row['DIAGNOSIS']==3):
        return 'AD'
    else:
        return 'not defined'


# Import csv files that contain label information
f_tab_adni_tau = pd.read_csv('./data/ADNI_Tau_Amyloid_SUVR_amyloid_tau_status_dems.csv', low_memory=False)
diagnosis_dxsum = pd.read_csv('/vol/chameleon/projects/adni/Diagnosis/DXSUM_PDXCONV_ADNIALL.csv', low_memory=False)
diagnosis_blchange = pd.read_csv('/vol/chameleon/projects/adni/Diagnosis/BLCHANGE.csv', low_memory=False)
diagnosis_adsx = pd.read_csv('/vol/chameleon/projects/adni/Diagnosis/ADSXLIST.csv', low_memory=False)




# open json that holds information about the split
with open('data_set_split.json', 'r') as f:
    dict_split = json.load(f)

# crate a csv file for each split 
for mode in ['train', 'val', 'test']:
    # retrieve the list of IDs for the current split
    list_ids = dict_split[mode]

    # init df that contains the paths to the image data 
    data_paths = pd.DataFrame(columns=['ID', 'ses', 'path_pet1451', 'path_anat', 'path_anat_mask', 'label'])
    not_available_pet1451 = 0
    not_available_anat = 0

    # iterate over all IDs in data_bids_processed
    for id in sorted(os.listdir("/vol/chameleon/projects/adni/data_bids_processed")):
        # check if patient ID is in the current split
        if id in list_ids:
            idpath = os.path.join("/vol/chameleon/projects/adni/data_bids_processed", id)
            # get a list with all the modalities that are available for the patient
            modalities = os.listdir(idpath)

            # check for all modalities
            
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
                            new_row = pd.Series({'ID': id, 'ses': session, 'path_pet1451': path_mni2mm, 'label': label})
                            # add the new row to the df
                            data_paths = pd.concat([data_paths, new_row.to_frame().T], ignore_index=True)
                        except:
                            not_available_pet1451 += 1

                        #print(data_paths)

                        

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
                        
                        if days < THRESHOLD_DAYS:
                            # extract data for relevant row
                            row_subject = df_subject.loc[index_subject]
                            # extract diagnosis
                            label = get_diag(row_subject)
                            
                            new_row = pd.Series({'ID': id, 'ses': session, 'path_anat': path_mni2mm, 'path_anat_mask': mask_path,'label': label})
                            # add the new row to the df that holds the paths
                            data_paths = pd.concat([data_paths, new_row.to_frame().T], ignore_index=True)
                        else:
                            not_available_anat += 1

                        
                        


    print(f'for {mode} split there are {not_available_pet1451} PET1451 and {not_available_anat} ANAT labels not available!')
    
    path_save = os.path.join(os.getcwd(), f'data/{mode}_path_data_labels.csv')
    
    data_paths.to_csv(path_save)
