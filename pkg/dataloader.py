import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import sys

################################
# TODO
#
# 1) always use all the features from tabular part? --> dropping by 'AGE' key not a nice solution


# ds = pd.read_csv('/vol/chameleon/projects/adni/Adni_merged.csv')
# filt  = ds['RID'] == 'sub-0021'
# print(ds.loc[47, 'ICV'])

# modalities: list

# how to merge sessions together?
# 1) check if len of modalities is 1 
#  --> then we can drop all other rows, do a reindex and that's it
# 2) choose a reference modality
#  --> go iteratively over other modalities
# for each session:
# go over remaining modalities


# goal: have a df with a subject id that holds the merged information from all modalities


# __getitem__ returns a dict:
# {'pet': None, 't1w': normalized tensor, 'tabular': innput feature vector}
# if df[pet] is not None:
#      dict['pet'] = ...
#
#     ....
#
# if df[tabular] is not None:  HERE WE NEED AN ADDITIONAL COLUMN THAT INDICATES WHETHER WE HAVE TAB DATA OR NOT
#       - extract columns
#       - fill in missing values
#       dict['tabular'] = feature vector




# tabular: list of all available feature cols in tabular

# TODO check missing values in dataframe

def merge_dfs(df_list):
    ref_df = df_list[0]
    #iterate over reference df

class MultiModalDataset(Dataset):
    
    def __init__(self, 
                path, 
                modalities=['pet1451', 't1w', 'tabular'],
                binary_classification=False,
                days_threshold=180
                ):
        """
        The constructor for MultiModalDataset class.
  
        Parameters:
           path (str): The path to the csv file that holds the paths for all modalities and all labels.   
        """
        self.modalities = modalities
        # read the dataframe with all modalities
        self.entire_ds = pd.read_csv(path)
        
        self.df_list = []
        # built reduced dataframe
        self.ds_combined = pd.DataFrame()
        if 'pet1451' in self.modalities:
            self.df_pet = self.entire_ds.dropna(subset='path_pet1451')
            self.df_list.append(self.df_pet)
            self.ds_combined = pd.concat([self.ds_combined, self.df_pet], ignore_index=True)

        if 't1w' in self.modalities:
            self.df_anat = self.entire_ds.dropna(subset='path_anat')
            self.df_list.append(self.df_anat)
            self.ds_combined = pd.concat([self.ds_combined, self.df_anat], ignore_index=True)

        if 'tabular' in self.modalities:
            self.df_tab = self.entire_ds.dropna(subset='AGE')
            self.df_list.append(self.df_tab)
            self.ds_combined = pd.concat([self.ds_combined, self.df_tab], ignore_index=True)
        
        # remove MCI labels if desired
        if binary_classification:
            self.ds_combined = self.ds_combined[self.ds_combined['label'] != 'MCI']


        # merge samples from different modalities with same label in a given timeintervals
        if len(self.modalities) == 1:
            self.ds = self.ds_combined
        
        else:
            self.ds = merge_dfs(self.df_list)
            

    def __len__(self):
        return len(self.ds)


label_mapping = {'CN': 0, 'MCI': 1, 'Dementia': 2}

class PETAV1451Dataset(Dataset):
    def __init__(self, path, transform=None, balanced=False):
        self.transform = transform
        self.ds = pd.read_csv(path)
        if balanced:
            self.ds = self.balance_ds()
            self.ds = self.ds.reset_index()
            

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        path_idx = self.ds.loc[index, 'path']
        im = nib.load(path_idx)
        data = im.get_fdata()
        if self.transform:
            data = self.transform(data)


        label = self.ds.loc[index, 'label']
        label = label_mapping[label]
        label = torch.tensor(label)
        return data, label
        #return super().__getitem__(index)

#TODO: use label mappings to ensure the correct weights are used
    def get_label_distribution(self):
        counts_normalized = self.ds['label'].value_counts(normalize=True)
        counts = self.ds['label'].value_counts()
        return torch.tensor(counts), torch.tensor(counts_normalized)

    def balance_ds(self):
        filt_ad = self.ds['label'] == 'Dementia'
        df_ad = self.ds.loc[filt_ad]
        ad_idx = df_ad.index
        num_samples = len(df_ad)

        filt_mci = self.ds['label'] == 'MCI'
        df_mci = self.ds.loc[filt_mci]
        df_mci = df_mci.sample(n=num_samples, random_state=1)
        mci_idx = df_mci.index
       
        filt_cn = self.ds['label'] == 'CN'
        df_cn = self.ds.loc[filt_cn]
        df_cn = df_cn.sample(n=num_samples, random_state=1)
        cn_idx = df_cn.index

        combined_idx = ad_idx.union(mci_idx).union(cn_idx)
        
        return self.ds.loc[combined_idx]

if __name__ == "__main__":
    path = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')
    dataset = MultiModalDataset(path=path)
    print(len(dataset))
    
    # x, y = dataset[3]
    # print(x.shape)
    # print(y)

    
