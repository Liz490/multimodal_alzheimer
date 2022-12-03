import pandas as pd
import json
import sys
import nibabel as nib
import numpy as np
import collections
diagnosis_dxsum = pd.read_csv('/vol/chameleon/projects/adni/Diagnosis/DXSUM_PDXCONV_ADNIALL.csv', low_memory=False)
diagnosis_blchange = pd.read_csv('/vol/chameleon/projects/adni/Diagnosis/BLCHANGE.csv', low_memory=False)
diagnosis_adsx = pd.read_csv('/vol/chameleon/projects/adni/Diagnosis/ADSXLIST.csv', low_memory=False)


def get_diag(row):
    if (row['DXCURREN']==1) or (row['DXCHANGE']==1) or (row['DXCHANGE']==7) or (row['DXCHANGE']==9) or (row['DIAGNOSIS']==1):
        return 'CN'
    elif (row['DXCURREN']==2) or (row['DXCHANGE']==2) or (row['DXCHANGE']==4) or (row['DXCHANGE']==8) or (row['DIAGNOSIS']==2):
        return 'MCI'
    elif (row['DXCURREN']==3) or (row['DXCHANGE']==3) or (row['DXCHANGE']==5) or (row['DXCHANGE']==6) or (row['DIAGNOSIS']==3):
        return 'AD'
    else:
        return 'not defined'


# USERDATE OR EXAMDATE?
# USERDATE: Date record created
# USERDATE2: Date record last updated
# EXAMDATE: Examination Date
sub_94 = diagnosis_dxsum.loc[diagnosis_dxsum['RID'] == 94]
print(sub_94[['ID', 'RID', 'USERDATE', 'USERDATE2', 'EXAMDATE', 'DXCURREN', 'DXCHANGE', 'DIAGNOSIS']])
sys.exit()

# Order of normalization correct???


# fr = diagnosis_dxsum.iloc[491]
# print(fr.loc[['DXCURREN', 'DXCHANGE', 'DIAGNOSIS']])
# print(get_diag(fr))

# 1. make a .csv file where you insert all train subjects
# 2. insert nan if a modality is not there
# 3. check a timeinterval for MRI data

# (176, 240, 256)
path = '/vol/chameleon/projects/adni/data_bids_processed/sub-0021/anat/ses-2010-10-08/sub-0021_ses-2010-10-08_T1w_N4_template0_reg_ants2_MNI_2mm.nii.gz'
#path = '/vol/chameleon/projects/adni/data_bids_processed/sub-0021/anat/template/sub-0021_T1w_N4_template0_ants2_MNI.nii.gz'
path_mask = '/vol/chameleon/projects/adni/data_bids_processed/sub-0021/anat/ses-2010-10-08/antsCorticalThickness/BrainExtractionMask_ants2_MNI_2mm.nii.gz'
#path_mask = '/vol/chameleon/projects/adni/data_bids_processed/sub-0021/anat/template/sub-0021_T1w_N4_template0.nii.gz'
im = nib.load(path)
im_mask = nib.load(path_mask)
data = im.get_fdata()
data_mask = im_mask.get_fdata()
#hist = np.histogram(data)
print(data.shape)
#hist_mask = np.histogram(data_mask)
print(data_mask.shape)
#counter = collections.Counter(data_mask)
#print(data_mask.numel())
#print(type(data_mask))
sys.exit()
#print(diagnosis['ID'])

#print(diagnosis_dxsum.columns)

#print(diagnosis[['RID', 'DXCHANGE', 'DXCURREN']])

sub_21 = diagnosis_dxsum.loc[diagnosis_dxsum['RID'] == 21]
print(sub_21[['RID', 'DIAGNOSIS', 'DXCURREN', 'DXCHANGE', 'USERDATE']])

#print(diagnosis.loc[diagnosis['RID'].idxmax()])


# ses-2010-10-08  ses-2011-10-13  ses-2012-10-04  ses-2013-10-17  ses-2015-11-03  ses-2018-01-25 

# ses-2018-01-25 flairs