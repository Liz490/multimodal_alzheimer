'''
Script to compute the z-score statistics over a dataset
'''
from dataloader import MultiModalDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
import os
import torch

transform = Compose([ToTensor()])
# we only use statistics of train set (also for validation and testing)
path = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')

# CHANGE arguments modalities and binary_classification depending on the modality and task
############
# MRI
# 2 targets
trainset_mri = MultiModalDataset(path=path, modalities=['t1w'], normalize_mri=None, binary_classification=True)
# 3 targets
# trainset_mri = MultiModalDataset(path=path, modalities=['t1w'], normalize_mri=None, binary_classification=False)
#############
# PET
# 2 targets
# trainset_pet = MultiModalDataset(path=path, modalities=['pet1451'], normalize_mri=None, binary_classification=True)
# 3 targets
# trainset_pet = MultiModalDataset(path=path, modalities=['pet1451'], normalize_mri=None, binary_classification=False)
#############

class NormalizeDataset():
    def __init__(self):
        pass
        
    @staticmethod
    def compute_std_mean(dataset, dim=None):

        
        loader = DataLoader(dataset, batch_size=1) 
        print(len(loader))
        mean_x, mean_x_squared = 0, 0
        if dim==None:
            print("Statistics are computed over all dimensions")
            for batch in loader:
                x = batch['mri']
                mean_x += x.mean()
                mean_x_squared += (x ** 2).mean()
        else:
            for x,_ in loader:
                mean_x += x.mean(dim=dim)
                mean_x_squared += (x ** 2).mean(dim=dim)


        mean = mean_x/len(loader)
        std = torch.sqrt((mean_x_squared/len(loader)) - mean ** 2) 

        return mean, std

print(NormalizeDataset.compute_std_mean(trainset_mri))
