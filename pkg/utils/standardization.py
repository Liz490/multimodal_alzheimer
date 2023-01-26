from dataloader import MultiModalDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
import os
import torch

transform = Compose([ToTensor()])
path = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')
#dataset = PETAV1451Dataset(path=path, transform=transform, balanced=False)  

# path_anat_train = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')
# path_anat_val = os.path.join(os.getcwd(), 'data/val_path_data_labels.csv')
# MAKE SURE TO DISTINGUISH BETWEEN TRAIN, VAL AND TEST
trainset_mri = MultiModalDataset(path=path, modalities=['t1w'], normalize_mri=None, binary_classification=True)

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
                #print((x**2).shape)
                # print(x)
                # print(x.mean())
                #break
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
