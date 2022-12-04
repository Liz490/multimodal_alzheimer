import os
import json
import pandas as pd

def writeTables(MCI = True):
    with open('../data_split_tabular_bio.json', 'r') as f:
        dict_split = json.load(f)

    adni_merged = pd.read_csv('../../Adni_merged.csv').dropna(subset = ['EXAMDATE', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV', 'AGE', 'DX'])

    for mode in ['train', 'val', 'test']:
        list_ids = dict_split[mode]
        df_tabular = pd.DataFrame(columns=['ID', 'ses', 'Ventricles', 'Hippocampus', 'Entorhinal', 'Fusiform', 'MidTemp', 'WholeBrain', 'age', 'label'])

        for subject in list_ids:
            rows_df = adni_merged.loc[adni_merged['RID'] == subject]
            #sort rows by examination date
            rows_df['EXAMDATE'] = pd.to_datetime(rows_df['EXAMDATE'], format='%d/%m/%Y')
            rows_df.sort_values('EXAMDATE', inplace= True)
            i = 0

            while i < rows_df.shape[0]-1:
                ventr = rows_df['Ventricles'].iloc[i]
                hippo = rows_df['Hippocampus'].iloc[i]
                wb = rows_df['WholeBrain'].iloc[i]
                ento = rows_df['Entorhinal'].iloc[i]
                fusi = rows_df['Fusiform'].iloc[i]
                midTemp = rows_df['MidTemp'].iloc[i]
                icv = rows_df['ICV'].iloc[i]
                age = rows_df['AGE'].iloc[i]
                label = rows_df['DX'].iloc[i]

                if MCI or label != 'MCI':
                    current_ses = rows_df['EXAMDATE'].iloc[i]
                    next_ses = rows_df['EXAMDATE'].iloc[i+1]

                    #if next session is within 6 months, it is considered 1 examination -> skip results from next session
                    if (next_ses.year - current_ses.year) * 12 + next_ses.month - current_ses.month < 6:
                        i += 1

                    new_row = {'ID': subject, 'ses': current_ses, 'Ventricles': ventr/icv ,
                               'Hippocampus': hippo/icv,
                               'Entorhinal': ento/icv, 'Fusiform': fusi/icv, 'MidTemp': midTemp/icv, 'WholeBrain': wb/icv, 'age': age, 'label': label}

                    df_tabular = df_tabular.append(new_row, ignore_index=True)
                i += 1

            if i < rows_df.shape[0] and (rows_df['DX'].iloc[i] != 'MCI' or MCI) :
                ventr = rows_df['Ventricles'].iloc[i]
                hippo = rows_df['Hippocampus'].iloc[i]
                wb = rows_df['WholeBrain'].iloc[i]
                ento = rows_df['Entorhinal'].iloc[i]
                fusi = rows_df['Fusiform'].iloc[i]
                midTemp = rows_df['MidTemp'].iloc[i]
                icv = rows_df['ICV'].iloc[i]
                age = rows_df['AGE'].iloc[i]
                label = rows_df['DX'].iloc[i]

                new_row = {'ID': subject, 'ses': rows_df['EXAMDATE'].iloc[i], 'Ventricles': ventr / icv,
                           'Hippocampus': hippo / icv,
                           'Entorhinal': ento / icv, 'Fusiform': fusi / icv, 'MidTemp': midTemp / icv,'WholeBrain': wb/icv,
                           'age': age, 'label': rows_df['DX'].iloc[i]}
                df_tabular = df_tabular.append(new_row, ignore_index=True)

        path_save = os.path.join(os.getcwd(), f'./{mode}_tabular_data_bio.csv')
        df_tabular.to_csv(path_save)

if __name__ == "__main__":
    writeTables(MCI=False)
    print('CSV files for each set was created!')