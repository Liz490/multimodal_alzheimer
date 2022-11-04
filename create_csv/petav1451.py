import os 
import glob
import pandas as pd

df_pet_av1451 = pd.DataFrame(columns=['ID', 'ses', 'path'])

for id in sorted(os.listdir("/vol/chameleon/projects/adni/data_bids_processed")):
    if id.startswith("sub"):
        filepath = os.path.join("/vol/chameleon/projects/adni/data_bids_processed", id)
        modalities = os.listdir(filepath)
        if 'pet-AV1451' in modalities:
            sessionspath = os.path.join(filepath, 'pet-AV1451')
            sessions = os.listdir(sessionspath)
            for session in sessions:
                path_ses = os.path.join(sessionspath, session)
                files = os.listdir(path_ses)
                relevant_files = [s for s in files if "MNI_2mm" in s]
                if len(relevant_files) == 1:
                    p = relevant_files[0]
                    p = os.path.join(path_ses, p)
                    new_row = {'ID': id, 'ses': session, 'path': p}
                    df_pet_av1451 = df_pet_av1451.append(new_row, ignore_index=True)


        # print(modalities)
        # new_row = {'ID': file, 'modalities': len(modalities)}
        # subject_modalities = subject_modalities.append(new_row, ignore_index=True)
df_pet_av1451.to_csv('/u/home/eisln/adlm_adni/data/path_data_petav1451.csv')