import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class PETAV1451Dataset(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        self.ds = pd.read_csv(path)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        path_idx = self.ds.loc[index, 'path']
        im = nib.load(path_idx)
        data = im.get_fdata()
        return data
        #return super().__getitem__(index)

if __name__ == "__main__":
    path = os.path.join(os.getcwd(), 'data/path_data_petav1451.csv')
    dataset = PETAV1451Dataset(path=path)
    print(dataset[0].shape)

    