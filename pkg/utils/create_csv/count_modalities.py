'''
In this file we create a csv to plot the distribution of the number of available image modalities 
over all patients (only patients with image modalites are considered).
'''
import os 
import glob
import pandas as pd

subject_modalities = pd.DataFrame(columns=['ID', 'anat', 'flair', 'pet-AV1451', 'pet-AV45', 'pet-FFB'])
subject_modalities = pd.DataFrame(columns=['ID', 'modalities'])

for file in sorted(os.listdir("/vol/chameleon/projects/adni/data_bids_processed")):
    if file.startswith("sub"):
        path = os.path.join("/vol/chameleon/projects/adni/data_bids_processed", file)
        modalities = os.listdir(path)
        new_row = {'ID': file, 'modalities': len(modalities)}
        subject_modalities = subject_modalities.append(new_row, ignore_index=True)
        
print(subject_modalities)

subject_modalities.to_csv('data/num_image_modalities.csv')