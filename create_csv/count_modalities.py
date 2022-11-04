import os 
import glob
import pandas as pd

subject_modalities = pd.DataFrame(columns=['ID', 'anat', 'flair', 'pet-AV1451', 'pet-AV45', 'pet-FFB'])
subject_modalities = pd.DataFrame(columns=['ID', 'modalities'])
# print(len(subject_modalities))
# subject_modalities = subject_modalities.append({'anat':1}, ignore_index=True) 
for file in sorted(os.listdir("/vol/chameleon/projects/adni/data_bids_processed")):
    if file.startswith("sub"):
        path = os.path.join("/vol/chameleon/projects/adni/data_bids_processed", file)
        modalities = os.listdir(path)
        print(modalities)
        new_row = {'ID': file, 'modalities': len(modalities)}
        subject_modalities = subject_modalities.append(new_row, ignore_index=True)
        # for mod in modalities:
        #     new_row[mod] = True
        # subject_modalities = subject_modalities.append(new_row, ignore_index=True)
        # break
# print(len(subject_modalities))

print(subject_modalities)

subject_modalities.to_csv('num_image_modalities.csv')