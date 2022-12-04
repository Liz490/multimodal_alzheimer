import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np


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
        counts_normalized = counts_normalized.reindex(
            index=['CN', 'MCI', 'Dementia'])
        counts = counts.reindex(index=['CN', 'MCI', 'Dementia'])
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
    path = os.path.join(os.getcwd(), 'data/path_data_petav1451.csv')
    dataset = PETAV1451Dataset(path=path, balanced=True)
    #print(len(dataset))
    # x, y = dataset[3]
    # print(x.shape)
    # print(y)

    
