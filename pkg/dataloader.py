import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from torchvision.transforms import Normalize


label_mapping = {'CN': 0, 'MCI': 1, 'Dementia': 2}
label_mapping_mri = {'CN': 0, 'MCI': 1, 'AD': 2}

class PETAV1451Dataset(Dataset):
    def __init__(self, path, transform=None, balanced=False):
        self.transform = transform
        self.ds = pd.read_csv(path)
        self.ds = self.ds.dropna(subset='path_pet1451')
        self.ds.reset_index(drop=True, inplace=True)
        
        if balanced:
            self.ds = self.balance_ds()
            self.ds = self.ds.reset_index()
            

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        path_idx = self.ds.loc[index, 'path_pet1451']
        
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

class AnatDataset(Dataset):
    def __init__(self, path, transform=None, normalization=False, subset=None, binary=False):
        self.transform = transform
        self.ds = pd.read_csv(path)
        # TODO: consider also dropping the values in case there is no mask available
        self.ds = self.ds.dropna(subset='path_anat')
        self.ds.reset_index(drop=True, inplace=True)
        self.normalization = normalization
        
        if subset:
            self.ds = self.ds.sample(n=subset, random_state=1)
            self.ds.reset_index(drop=True, inplace=True)
            
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        path_idx = self.ds.loc[index, 'path_anat']
        path_mask = self.ds.loc[index, 'path_anat_mask']

        # image
        im = nib.load(path_idx)
        data = im.get_fdata()
        data = torch.tensor(data)
        # print('min and max')
        # print(data.min())
        # print(data.max())
        
        # mask
        mask = nib.load(path_mask)
        binary_mask = mask.get_fdata()
        binary_mask = torch.tensor(binary_mask)

        # compute mean and variance wrt brain mask
        # 1. set non-brain voxels to 0 
        data_masked = data * binary_mask
        
        # 2. flatten the tensor and remove all zero entries
        data_masked = data_masked.reshape(-1)
        data_masked = data_masked[data_masked.nonzero()]
        
        # 3. compute mean and std
        std_mask, mean_mask = torch.std_mean(data_masked)
        



        if self.transform:
            data = self.transform(data)

        if self.normalization:
            normalization_mask = Normalize(mean=mean_mask, std=std_mask)
            data = normalization_mask(data)


        # apply mask again
        data *= binary_mask

        label = self.ds.loc[index, 'label']
        label = label_mapping[label]
        label = torch.tensor(label)
        return data, label
    
    def get_label_distribution(self):
        counts_normalized = self.ds['label'].value_counts(normalize=True)
        counts_normalized = counts_normalized.reindex(index = ['CN','MCI','Dementia'])
        counts = self.ds['label'].value_counts()
        counts = counts.reindex(index = ['CN','MCI','Dementia'])
        
        return torch.tensor(counts), torch.tensor(counts_normalized)
        

if __name__ == "__main__":
    path = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')
    dataset = AnatDataset(path=path, subset=15)
    print(len(dataset))
    # x, y = dataset[3]
    # print('min and max after normalization:')
    # print(x.min())
    # print(x.max())
    
    # print(x.shape)
    # print(y)

    # dataset = AnatDataset(path=path, normalization=False)
    # print(len(dataset))
    # x, y = dataset[3]
    # print('min and max without normalization:')
    # print(x.min())
    # print(x.max())
