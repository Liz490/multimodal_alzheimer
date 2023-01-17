from sqlite3 import Timestamp
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
from typing import List, Dict, Any, Callable, Tuple

pd.options.mode.chained_assignment = None

# transforms
# WATCH OUT THIS PERMUTES THE DIMENSIONS OF THE TENSOR
#to_tensor = ToTensor()

class MultiModalDataset(Dataset):
    """Dataset class that supports multi-modal training for PET, MRI and tabular data.

    It is possible to pass up to three modalities and samples are generated for all possible combinations 
    that lie within a certain time interval. Depending on the modality different normalization techniques
    can be applied such as min-max normalization, min-max normalization based on quantiles, z-score normalization, 
    etc. For the MRI data, it is possible to apply per-scan normalization based on a brain mask where only brain 
    voxels are considered. The MultiModalDataset supports 3-class (CN, MCI, Dementia) and binary classification 
    (CN, Dementia). In addition, weights for the loss function can be computed to account for class imbalances.

    Attributes:
        path: A string indicating the path to the .csv file containing data and labels
        binary_classification: A boolean that indicates whether the MCI class is dropped
            or not
        modalities: A list of strings containing the names of the modalities that are 
            desired to be used. Possible strings: 'pet1451', 't1w', 'tabular'
        days_threshold: An integer indicating how many days can be at most between the 
            acquisition of data from different modalities to still merge them to the 
            same sample.
        transform_pet: ...
        transform_mri: ...
        transform_tabular= ...
        normalize_pet: 
            A dict with the keys 'std' and 'mean' that correspond to the statistics over the entire split of the PET data
            or 
            None
        normalize_mri: 
            A dict of length 1 with the key 'per_scan_norm' with 
                the value 'normalize' if you want to standardize the MRI images per scan only based on the brain voxels
                or 
                with the value 'min_max' if you want to map the MRI images into the range [0,1] only based on the brain 
                voxels
            or 
            A dict of length 1 with the key 'all_scan_norm' where the value is again a dict with the keys 'std' and 'mean' 
            that correspond to the statistics over the entire split of the MRI data
            or 
            None
        quantile: A float that indicates for which percentile the per scan min_max normalization is performed
                    e.g: quantile=0.99 means that the miniumum value is mapped to the 1 percentile and the maximum is
                    mapped to the 99 percentile.
    """
    
    def __init__(self, 
                path: str,
                binary_classification: bool=False,
                modalities: List[str]=['pet1451', 't1w', 'tabular'],
                days_threshold: int=180,
                transform_pet=None,
                transform_mri=None,
                transform_tabular=None,
                normalize_pet: Dict[str, float]=None,
                normalize_mri: Dict[str, Any]=None,
                quantile: float=0.99, 
                ):
        """
        Constructor for the MultiModalDataset class.
        """
        
        # read the dataframe with all modalities
        self.entire_ds = pd.read_csv(path)

        # remove MCI labels if desired
        self.binary_classification = binary_classification
        if self.binary_classification:
            self.entire_ds = self.entire_ds[self.entire_ds['label'] != 'MCI']
            self.label_mapping = {'CN': 0, 'Dementia': 1}
        else:
            self.label_mapping = {'CN': 0, 'MCI': 1, 'Dementia': 2}
        
        self.days_threshold = days_threshold
        # list that will hold dataframes for each individual modality
        self.df_list = []

        self.modalities = modalities
        # self.modality list contains only correct entries and each modality only once
        assert(len(self.modalities) in range(1,4))
        assert(all([x in ['pet1451', 't1w', 'tabular'] for x in self.modalities]))
        assert(len(set(self.modalities)) == len(self.modalities))
        
        # built reduced dataframe depending on entries in self.modalities
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
            # self.ds is the dataframe that is being used in the end
            self.ds = self.ds_combined
        else:
            for ds in self.df_list:
                ds['ses'] = ds['ses'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
            # 1) choose a base dataframe
            # 2) go over every row 
            # 3) take the dataframe of the next modality and find all matches
            # 4) append all matched samples to a new dataframe
            # 5) this new dataframe is the base dataframe and we go back to step 2)
            df_base = self.df_list[0]
            # min_time and max_time are used to ensure that new samples are only merged if both are
            # at most days_threshold away
            df_base['min_time'] = df_base['ses']
            df_base['max_time'] = df_base['ses']
            df_base = df_base.drop(columns='ses')
            for i in range((len(self.df_list) - 1)):
                df_result_i = pd.DataFrame()
                for idx, row in df_base.iterrows():
                    df_i = self.df_list[i+1]
                    df_matching = find_corresponding_samples(df=df_i,
                                                            id=row['ID'],
                                                            label=row['label'],
                                                            min_time=row['min_time'],
                                                            max_time=row['max_time'],
                                                            max_days=self.days_threshold,
                                                            )
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
            assert isinstance(self.normalize_pet['mean'], float) 
            assert 'std' in self.normalize_pet.keys()
            assert isinstance(self.normalize_pet['std'], float) 
            #(self.normalize_pet.contains_key())
        self.normalize_mri = normalize_mri
        self.quantile = quantile
        


    def __len__(self) -> int:
        """
        Compute the length of the dataset
        """
        return len(self.ds)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Build a dictionary of data label pairs depending on the desired modalities

        Args:
        index: An integer indicating the index of the sample

        Returns:
        A dictionary with the following keys:
            'label': A torch.tensor according to the label mapping
            'pet1451': A torch.tensor with the PET data or None if not available
            'mri': A torch.tensor with the MRI data or None if not available
            'tabular': A torch.tensor with the tabular data or None if not available
        """
        data = {}
        sample = self.ds.iloc[index]
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
        # load data and mask
        path_mri = sample['path_anat']
        path_mri_mask = sample['path_anat_mask']
        if path_mri == None:
            mri_data = None
        else:
            mri_im = nib.load(path_mri)
            mri_data = mri_im.get_fdata()
        
            if self.transform_mri:
                mri_data = self.transform_mri(mri_data)
    
            mri_data = torch.tensor(mri_data)

            if self.normalize_mri:
                assert isinstance(self.normalize_mri, dict)
                assert len(self.normalize_mri) == 1
                if 'per_scan_norm' in self.normalize_mri:
                    mri_mask = nib.load(path_mri_mask)
                    binary_mask_mri = mri_mask.get_fdata()
                    binary_mask_mri = torch.tensor(binary_mask_mri)

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
                    elif self.normalize_mri['per_scan_norm'] == 'min_max':
                        assert self.quantile >= 0 and self.quantile <= 1
                        quant_max = torch.quantile(data_masked_mri, self.quantile, interpolation='linear')
                        quant_min = torch.quantile(data_masked_mri, 1-self.quantile, interpolation='linear')

                        mri_data = (mri_data - quant_min) / (quant_max - quant_min)
                        mri_data[mri_data > 1] = 1
                        mri_data[mri_data < 0] = 0

                        mri_data *= binary_mask_mri
                    else:
                        raise ValueError('If you want to normalize per scan you have to pass either "normalize" or "min_max"')

                elif 'all_scan_norm' in self.normalize_mri:
                    assert 'mean' in self.normalize_mri['all_scan_norm'].keys()
                    assert 'std' in self.normalize_mri['all_scan_norm'].keys()
                    all_scan_normalize_mri = Normalize(mean=self.normalize_mri['all_scan_norm']['mean'], std=self.normalize_mri['all_scan_norm']['std'])
                    mri_data = all_scan_normalize_mri(mri_data)

                else:
                    raise ValueError('If you use the argument "normalize_mri" only "per_scan_norm" or "all_scan_norm" are allowed as keys!')
       

        data['mri'] = mri_data

        #########################################################################################################
        # STILL TODO!!!!!
        ###########
        # TABULAR #
        ########### 

        if sample['AGE'] == None:
            tabular_data = None
        else:
            # ICV
            icv = sample['ICV']
            # features normalized 
            age = sample['AGE']
            pteducat = sample['PTEDUCAT']
            ventr = sample['Ventricles']
            hippocamp = sample['Hippocampus']
            whole_brain = sample['PTEDUCAT']
            entorhinal = sample['Entorhinal']
            fusiform = sample['Fusiform']
            mid_temp = sample['MidTemp']

            tabular_data = torch.tensor([age, pteducat, ventr, hippocamp, whole_brain, entorhinal, fusiform, mid_temp, icv])

        data['tabular'] = tabular_data
        # STILL TODO!!!!!
        #########################################################################################################

        #########
        # LABEL #
        #########
        label = self.label_mapping[sample['label']]
        data['label'] = torch.tensor(label)

        # before we return the dict we have to remove all None keys because
        # the collate_fn function in the pytorch dataloader tries to collate each individual key to a batch
        # and if there is a None key it does not know how to handle it
        data = {k: v for k, v in data.items() if v is not None}
        return data

    def get_label_distribution(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the frequency of class ocurrences

        Returns:
        A tuple of torch tensors with:
            the absolute frequency of each class occurance
            and
            the normalized class occurances such that they add up to 1
        """
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


def find_corresponding_samples(df: pd.DataFrame,
                            id: str,
                            label: str,
                            min_time: pd.Timestamp,
                            max_time: pd.Timestamp,
                            max_days: int=180,
) -> pd.DataFrame:
    """
    Finds all samples in a DataFrame that can be fused with another sample.

    This sample is characterized by the remaining arguments of the function

    Args:
    df: A dataframe in which corresponding samples are searched based on several conditions
    id: A string that indicates the ID the dataframe is filtered by
    label: A string that indicates the label the dataframe is filtered by
    min_time: A timestamp that indicates the earliest timestamp of all datasamples that have been
        fused so far
    max_time: A timestamp that indicates the latest timestamp of all datasamples that have been
        fused so far
    max_days: the maximum number of days that all samples that need to be fused can be apart from
        each other

    Returns:
    A reduced dataframe with all samples that can be fused

    Example:
        We have already fused a sample of the MRI modality with one of the PET modality. MRI data was
        acquired on 2018-02-02 (min_time) and PET data was obtained on 2018-01-25 (max_time). Next, the
        dataframe that contains tabular data is queried for data with the same ID and the same label. 
        Among all these samples we try to fuse the data such that the date of the new sample, min_time and
        max_time are at most max_days (e.g: 180) apart from each other.
    """
    filt_id_label = (df['ID'] == id) & (df['label'] == label)
    df = df.loc[filt_id_label]
    if len(df) == 0:
        return df
    else:
        # get timedelta differences in days
        df['min'] = df['ses'] - min_time
        df['min'] = df['min'].apply(lambda x: x.days)
        df['max'] = max_time - df['ses'] 
        df['max'] = df['max'].apply(lambda x: x.days)
        # 1) df['ses'] - min_time is positive if df['ses'] is larger and should be at most max_days away
        # 2) max_time - df['ses']  is positive if df['ses'] is smaller and should be at most max_days away
        filt_time_diff = (df['min'] <= max_days) & (df['max'] <= max_days)
        df = df.loc[filt_time_diff]
        df = df.drop(columns=['min', 'max'])
        df = df.reset_index(drop=True)
        return df

def merge_two_dfs(df1: pd.Series, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Combine data from two dataframes.

    Args:
    df1: A Series representing a sample of the modalities that have already been fused
    df2: A DataFrame representing all samples of the new modality that can 
        be fused with the sample in df1

    Returns:
    A DataFrame that holds the combined information of all modalities that have been fused so far

    Example:
        df1 is a Series of a sample with only PET data, thus it has NaN data for all columns that 
        are related to MRI or tabular data. df2 is a DataFrame of samples with MRI data that can 
        be merged to the sample of PET data. Thus df2 has data for 'path_anat' or 'path_anat_mask' 
        but NaN values in the column 'path_pet1451'. Now, df2 is checked for NaN columns and if 
        there is data available in df1 for that column, it is written to df2 which will then be the 
        merged dataframe.
    """
    # update 'min_time' and 'max_time' of the merged dataframe
    df2['min'] = df1['min_time'] - df2['ses']
    df2['min'] = df2['min'].apply(lambda x: x.days > 0)
    df2['max'] = df1['max_time'] - df2['ses']
    df2['max'] = df2['max'].apply(lambda x: x.days < 0)
    df2['min_time'] = df1['min_time']
    df2['min_time'] = df2.apply(lambda x: x['ses'] if x['min'] else x['min_time'], axis=1)
    df2['max_time'] = df1['max_time']
    df2['max_time'] = df2.apply(lambda x: x['ses'] if x['max'] else x['max_time'], axis=1)

    df2 = df2.drop(columns=['min', 'max', 'ses'])

    # if df2 has NaN for PET but df1 has PET data, it is written in the corresponding column of df2
    df1_nan = df1.isna()
    for col2 in df2.columns:
        if df2[col2].isnull().values.any():
            if df1_nan[col2] == False: 
                df2[col2] = df1[col2]
    return df2


    def remove_mci(self):
        self.ds = self.ds.loc[self.ds['label'] != 'MCI']

if __name__ == "__main__":
    trainpath = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')
    valpath = os.path.join(os.getcwd(), 'data/val_path_data_labels.csv')

    # how to use for PET
    normalization_pet = {'mean': 0.5145, 'std': 0.5383}
    trainset_pet = MultiModalDataset(path=trainpath, modalities=['pet1451'], normalize_pet=normalization_pet)
    data_label_pet = trainset_pet[0]
    print(data_label_pet['pet1451'].shape)
    print(data_label_pet['label'])

    # how to use for MRI
    normalization_mri = {'per_scan_norm': 'min_max'}
    trainset_mri = MultiModalDataset(path=trainpath, modalities=['t1w'], normalize_mri=normalization_mri, quantile=0.97)
    data_label_mri = trainset_mri[0]
    print(data_label_mri['mri'].shape)
    print(data_label_mri['label'])

    # how to use for tabular
    trainset_tabular = MultiModalDataset(path=trainpath, modalities=['tabular'])
    data_label_tabular = trainset_tabular[0]
    print(data_label_tabular['tabular'])
    print(data_label_tabular['label'])



    # # normalize_pet: None or dict with 'mean' and 'std'
    # std_all_scans = {'mean': 9.8, 'std': 8.2}
    # norm_mri = {'all_scan_norm': std_all_scans}
    # path = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')
    # dataset = MultiModalDataset(path=path, modalities=['t1w'], normalize_mri=norm_mri)
    # print(len(dataset))
    # d = dataset[6]
    # print(d['pet1451'].shape)
    # print(d['mri'].shape)
    # print(d['tabular'])
    # print(d['label'])
    # # x, y = dataset[3]
    # # print(x.shape)
    # # print(y)
    # mri_data = d['mri']
    # # print(mri_data[55,55,55])
    # # print(mri_data.min())
    # # print(mri_data.max())
    # # bins = 100
    # # hist = torch.histc(mri_data, bins=bins, min=0, max=1)
    # sns.histplot(mri_data.reshape(-1).numpy(), bins=20)
    # plt.savefig('tst')
    # # x = range(0,1,0.01)
    # # plt.bar(x, hist, align='center')
    # # plt.xlabel('Bins)
    # # plt.show()
