import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
import sys
from torchvision.transforms import ToTensor, Normalize
import matplotlib.pyplot as plt
import seaborn as sns

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


# transforms
# WATCH OUT THIS PERMUTES THE DIMENSIONS OF THE TENSOR
#to_tensor = ToTensor()

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
                **normalize_mri  # dict with mean and std or 'min_max' (additional percentil arg) or per_scan
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
        self.binary_classification = binary_classification
        if self.binary_classification:
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
        if self.normalize_pet:
            assert 'mean' in self.normalize_pet.keys()
            assert 'std' in self.normalize_pet.keys()
            #(self.normalize_pet.contains_key())
        self.normalize_mri = normalize_mri
        # print(self.normalize_mri)

        if len(self.normalize_mri) > 1:
            print('WARNING: YOU ARE NORMALIZING THE DATA MULTIPLE TIMES! PLEASE ONLY PROVIDE AT MOST ONE KEYWORD ARGUMENT')
        # if 'per_scan_norm' in self.normalize_mri:
        #     print('norm')
        #     print(self.normalize_mri['per_scan_norm'])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        data = {}
        sample = self.ds.iloc[index]
        # print(sample)
        ########
        # PET  #
        ########
        pet_path = sample['path_pet1451']
        if pet_path == None:
            pet_data = None
        else:
            pet_im = nib.load(pet_path)
            pet_data = pet_im.get_fdata()
            if self.transform_pet:
                pet_data = self.transform_pet(pet_data)
            
            pet_data = torch.tensor(pet_data)

            if self.normalize_pet:
                normalize_pet = Normalize(mean=self.normalize_pet['mean'], std=self.normalize_pet['std'])
                pet_data = normalize_pet(pet_data)

        data['pet1451'] = pet_data

        ###########
        # MRI T1W #
        ###########

        path_mri = sample['path_anat']
        path_mri_mask = sample['path_anat_mask']
        # print(path_mri)
        # print(path_mri_mask)
        if path_mri == None:
            mri_data = None
        else:
            mri_im = nib.load(path_mri)
            mri_data = mri_im.get_fdata()
            # sns.histplot(mri_data.reshape(-1))
            # plt.savefig('a')
            
            
            if self.transform_mri:
                mri_data = self.transform_mri(mri_data)
            #print(mri_data.shape)
            mri_data = torch.tensor(mri_data)

            if 'per_scan_norm' in self.normalize_mri:
                mri_mask = nib.load(path_mri_mask)
                binary_mask_mri = mri_mask.get_fdata()
                binary_mask_mri = torch.tensor(binary_mask_mri)

                # print(mri_data.shape)
                # print(binary_mask_mri.shape)
                # 1. set non-brain voxels to 0 
                data_masked_mri = mri_data * binary_mask_mri
        
                # 2. flatten the tensor and remove all zero entries
                data_masked_mri = data_masked_mri.reshape(-1)
                data_masked_mri = data_masked_mri[data_masked_mri.nonzero()]
                

                if self.normalize_mri['per_scan_norm'] == 'normalize':
                    # 3. compute mean and std
                    std_mask_mri, mean_mask_mri = torch.std_mean(data_masked_mri)

                    normalize_per_scan_mri = Normalize(mean=mean_mask_mri, std=std_mask_mri,)
                    mri_data = normalize_per_scan_mri(mri_data)

                    # 4. Multiply normalized data again with brain mask
                    mri_data *= binary_mask_mri
                else:
                    q = 0.99
                    quant_max = torch.quantile(data_masked_mri, q, interpolation='linear')
                    quant_min = torch.quantile(data_masked_mri, 1-q, interpolation='linear')

                    mri_data = (mri_data - quant_min) / (quant_max - quant_min)
                    mri_data[mri_data > 1] = 1
                    mri_data[mri_data < 0] = 0

                    mri_data *= binary_mask_mri

            elif 'all_scan_norm' in self.normalize_mri:
                assert 'mean' in self.normalize_mri['all_scan_norm'].keys()
                assert 'std' in self.normalize_mri['all_scan_norm'].keys()
                normalize_mri = Normalize(mean=self.normalize_mri['all_scan_norm']['mean'], std=self.normalize_pet['all_scan_norm']['std'])
                pet_data = normalize_pet(pet_data)

                
       

        data['mri'] = mri_data

        #########################################################################################################
        # STILL TODO!!!!!
        ###########
        # TABULAR #
        ########### 
        age = sample['AGE']
        if age == None:
            tabular_data = None
        else:
            tabular_data = torch.tensor(5)

        data['tabular'] = tabular_data
        # STILL TODO!!!!!
        #########################################################################################################

        #########
        # LABEL #
        #########
        label = self.label_mapping[sample['label']]
        data['label'] = torch.tensor(label)

        #'pet1451', 't1w', 'tabular'
        data = {k: v for k, v in data.items() if v is not None}
        return data

    def get_label_distribution(self):
        counts_normalized = self.ds['label'].value_counts(normalize=True)
        if self.binary_classification:
            counts_normalized = counts_normalized.reindex(index = ['CN', 'Dementia'])
        else:
            counts_normalized = counts_normalized.reindex(index = ['CN','MCI','Dementia'])
        counts = self.ds['label'].value_counts()
        if self.binary_classification:
            counts = counts.reindex(index = ['CN', 'Dementia'])
        else:
            counts = counts.reindex(index = ['CN','MCI','Dementia'])
        
        return torch.tensor(counts), torch.tensor(counts_normalized)

class PETAV1451Dataset(Dataset):
    def __init__(self, path, transform=None, balanced=False, remove_mci=False):
        self.label_mapping = {'CN': 0, 'MCI': 1, 'Dementia': 2}
        self.remove_mci_state = remove_mci
        self.transform = transform
        self.ds = pd.read_csv(path)
        if remove_mci:
            self.label_mapping = {'CN': 0, 'Dementia': 1}
            self.remove_mci()
            self.ds = self.ds.reset_index()
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
        label = self.label_mapping[label]
        label = torch.tensor(label)
        return data, label
        #return super().__getitem__(index)

#TODO: use label mappings to ensure the correct weights are used
    def get_label_distribution(self):
        counts_normalized = self.ds['label'].value_counts(normalize=True)
        counts = self.ds['label'].value_counts()
        if not self.remove_mci_state:
            counts_normalized = counts_normalized.reindex(
                index=['CN', 'MCI', 'Dementia'])
            counts = counts.reindex(index=['CN', 'MCI', 'Dementia'])
        else:
            counts_normalized = counts_normalized.reindex(
                index=['CN', 'Dementia'])
            counts = counts.reindex(index=['CN', 'Dementia'])
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

    def remove_mci(self):
        self.ds = self.ds.loc[self.ds['label'] != 'MCI']

if __name__ == "__main__":


    # normalize_pet: None or dict with 'mean' and 'std'

    norm_mri = 3
    path = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')
    dataset = MultiModalDataset(path=path, modalities=['pet1451', 't1w'], per_scan_norm='min_max')
    print(len(dataset))
    d = dataset[2]
    print(d['pet1451'].shape)
    print(d['mri'].shape)
    print(d['tabular'])
    print(d['label'])
    # x, y = dataset[3]
    # print(x.shape)
    # print(y)
    mri_data = d['mri']
    print(mri_data[55,55,55])
    # print(mri_data.min())
    # print(mri_data.max())
    # bins = 100
    # hist = torch.histc(mri_data, bins=bins, min=0, max=1)
    sns.histplot(mri_data.reshape(-1).numpy(), bins=20)
    plt.savefig('a')
    # x = range(0,1,0.01)
    # plt.bar(x, hist, align='center')
    # plt.xlabel('Bins)
    # plt.show()

