import os
import json
import pandas as pd

def writeTables():
    with open('data_split_tabular_schaefer.json', 'r') as f:
        dict_split = json.load(f)

    schaefer_regions = pd.read_csv('../../ADNI_Tau_Amyloid_SUVR_amyloid_tau_status_dems.csv')
    schaefer_regions = schaefer_regions[schaefer_regions.columns.drop(list(schaefer_regions.filter(regex='SUVR.DKT.ROI.idx.')))]
    #print(list[schaefer_regions.columns])
    schaefer_regions.drop(['sex', 'pet.modality', 'modality.description',
                           'image.id', 'ses', 't.diff.uwn.psych.pet.yrs', 'ADAS13', 'MMSE', 'mPACCtrailsB',
                           't.diff.adas.pet.yrs', 'PHASE', 'SITEID',
                           't.diff.diagnosis.pet.yrs', 'Phase', 'tau.braak56.SUVR', 'tau.temporal.meta.SUVR',
                           'tau.global.SUVR', 'RID',
                           'PTEDUCAT', 'ADNI_MEM', 'ADNI_EF', 'ADNI_LAN', 'ADNI_VS', 'amyloid.global.SUVR',
                           'amyloid.global.SUVR.status', 'centiloid',
                           'tau.braak1.SUVR', 'tau.braak3.SUVR', 'tau.braak4.SUVR',
                           'tau.braak5.SUVR', 'tau.braak6.SUVR', 'tau.braak34.SUVR', 'apoe', 'group'], axis=1, inplace=True)
    schaefer_regions = schaefer_regions.dropna()
    for mode in ['train', 'val', 'test']:
        list_ids = dict_split[mode]
        df_tabular = pd.DataFrame()
        for subject in list_ids:
            rows_df = schaefer_regions.loc[schaefer_regions['ID'] == subject]
            #sort rows by examination date
            rows_df['acq.date'] = pd.to_datetime(rows_df['acq.date'], format='%Y-%m-%d')
            rows_df.sort_values('acq.date', inplace= True)
            i = 0
            while i < rows_df.shape[0]-1:
                current_ses = rows_df['acq.date'].iloc[i]
                next_ses = rows_df['acq.date'].iloc[i+1]

                #if next session is within 6 months, it is considered 1 examination -> skip results from next session
                if (next_ses.year - current_ses.year) * 12 + next_ses.month - current_ses.month < 6:
                    i += 1

                df_tabular = df_tabular.append(rows_df.iloc[i], ignore_index=True)
                i += 1

            if i < rows_df.shape[0]:

                df_tabular = df_tabular.append(rows_df.iloc[i], ignore_index=True)

        path_save = os.path.join(os.getcwd(), f'./{mode}_tabular_data_schaefer.csv')
        df_tabular.drop(['ID','acq.date'], axis=1, inplace=True)
        print(df_tabular.shape)
        df_tabular.to_csv(path_save)

if __name__ == "__main__":
    writeTables()
    print('CSV files for each set was created!')