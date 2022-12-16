import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
import sys

pd.options.mode.chained_assignment = None


def find_corresponding_samples(df, id, label, min_time, max_time, threshold_days=180):
        
    # get subdataframe with same id
    filt_id_label = (df['ID'] == id) & (df['label'] == label)
    df = df.loc[filt_id_label]

    if len(df) == 0:
        return df
    else:
    
    # get timedelta differences
        df['min'] = df['ses'] - min_time
        df['min'] = df['min'].apply(lambda x: x.days)
        df['max'] = max_time - df['ses'] 
        df['max'] = df['max'].apply(lambda x: x.days)
        filt_time_diff = (df['min'] <= threshold_days) & (df['max'] <= threshold_days)
        df = df.loc[filt_time_diff]
        df = df.drop(columns=['min', 'max'])
        df = df.reset_index(drop=True)
        return df

def merge_two_dfs(df1, df2):
    # df1 is always a row, df2 can be a df of several rows
    df2['min'] = df1['min_time'] - df2['ses']
    df2['min'] = df2['min'].apply(lambda x: x.days > 0)
    df2['max'] = df1['max_time'] - df2['ses']
    df2['max'] = df2['max'].apply(lambda x: x.days < 0)
    
    df2['min_time'] = df1['min_time']
    df2['min_time'] = df2.apply(lambda x: x['ses'] if x['min'] else x['min_time'], axis=1)
    df2['max_time'] = df1['max_time']
    df2['max_time'] = df2.apply(lambda x: x['ses'] if x['max'] else x['max_time'], axis=1)

    df2 = df2.drop(columns=['min', 'max', 'ses'])

    df1_nan = df1.isna()
    for col2 in df2.columns:
        if df2[col2].isnull().values.any():
            if df1_nan[col2] == False: 
                df2[col2] = df1[col2]
    return df2


class MultiModalDataset(Dataset):
    
    def __init__(self, 
                path, 
                modalities=['pet1451', 't1w', 'tabular'],
                binary_classification=False,
                days_threshold=180,
                transform_pet=None,
                transform_mri=None,
                transform_tabular=None,
                normalize_pet=None,
                normalize_mri=None
                ):
        """
        The constructor for MultiModalDataset class.
  
        Parameters:
           path (str): The path to the csv file that holds the paths for all modalities and all labels.   
        """
        self.modalities = modalities
        # read the dataframe with all modalities
        self.entire_ds = pd.read_csv(path)

        # remove MCI labels if desired
        if binary_classification:
            self.entire_ds = self.entire_ds[self.entire_ds['label'] != 'MCI']
            self.label_mapping = {'CN': 0, 'Dementia': 1}
        else:
            self.label_mapping = {'CN': 0, 'MCI': 1, 'Dementia': 2}
        
        self.df_list = []
        # built reduced dataframe
        self.ds_combined = pd.DataFrame()
        if 'pet1451' in self.modalities:
            self.df_pet = self.entire_ds.dropna(subset='path_pet1451').reset_index(drop=True)
            self.df_list.append(self.df_pet)
            self.ds_combined = pd.concat([self.ds_combined, self.df_pet], ignore_index=True)

        if 't1w' in self.modalities:
            self.df_anat = self.entire_ds.dropna(subset='path_anat').reset_index(drop=True)
            self.df_list.append(self.df_anat)
            self.ds_combined = pd.concat([self.ds_combined, self.df_anat], ignore_index=True)

        if 'tabular' in self.modalities:
            self.df_tab = self.entire_ds.dropna(subset='AGE').reset_index(drop=True)
            self.df_list.append(self.df_tab)
            self.ds_combined = pd.concat([self.ds_combined, self.df_tab], ignore_index=True)
        
    


        # merge samples from different modalities with same label in a given timeintervals
        if len(self.df_list) == 1:
            self.ds = self.ds_combined
        
        else:
            # df_test = pd.read_csv('data/train_path_data_labels.csv', low_memory=False, index_col=0)
            for ds in self.df_list:
                ds['ses'] = ds['ses'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))

            df_base = self.df_list[0]
            df_base['min_time'] = df_base['ses']
            df_base['max_time'] = df_base['ses']
            df_base = df_base.drop(columns='ses')
            for i in range((len(self.df_list) - 1)):
                df_result_i = pd.DataFrame()
                for idx, row in df_base.iterrows():
                    df_i = self.df_list[i+1]
                    df_matching = find_corresponding_samples(df_i, row['ID'], row['label'], row['min_time'], row['max_time'])
                    # if len(df_matching) >= 2:
                    #     print('len')
                    #     print(len(df_matching))
                    #     print(row)
                    #     print(df_matching)
                    # print(row)
                    # print(df_matching)
                    if len(df_matching) >= 1:
                        df_merged = merge_two_dfs(row, df_matching)
                        df_result_i = pd.concat([df_result_i, df_merged], ignore_index=True)
                df_base = df_result_i.copy()
            self.ds = df_result_i

        self.ds = self.ds.replace({np.nan: None})

        # transforms
        self.transform_pet = transform_pet 
        self.transform_mri = transform_mri
        self.transform_tabular = transform_tabular  
        # normalization
        self.normalize_pet = normalize_pet
        self.normalize_mri = normalize_mri

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        data = {}
        sample = self.ds.iloc[index]
        print(sample)
        ########
        # PET #
        ########
        pet_path = sample['path_pet1451']
        if pet_path == None:
            pet_data = None
        else:
            pet_im = nib.load(pet_path)
            pet_data = pet_im.get_fdata()
        if self.transform_pet:
            pet_data = self.transform_pet(pet_data)
        data['pet1451'] = pet_data

        # TODO: if self.normalize

        #########
        # LABEL #
        #########
        data['label'] = self.label_mapping[sample['label']]

        #'pet1451', 't1w', 'tabular'

        return data

    
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
    dataset = MultiModalDataset(path=path, modalities=['pet1451'])
    print(len(dataset))
    d = dataset[1]
    print(d['pet1451'].shape)
    print(d['label'])
    # x, y = dataset[3]
    # print(x.shape)
    # print(y)

    
