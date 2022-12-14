import os
import json
import pandas as pd

def writeTables(MCI = True):
    with open('../data_split_tabular_bio.json', 'r') as f:
        dict_split = json.load(f)

    adni_merged = pd.read_csv('../../Adni_merged.csv').dropna(subset = ['EXAMDATE', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV', 'AGE', 'PTEDUCAT', 'DX'])

    for mode in ['train', 'val', 'test']:
        list_ids = dict_split[mode]

        df_tabular = create_csv(adni_merged= adni_merged, list_ids = list_ids, MCI = MCI)

        path_save = os.path.join(os.getcwd(), f'./{mode}_tabular_data_bio.csv')
        df_tabular.to_csv(path_save)


def create_csv(adni_merged, list_ids, MCI = True, extract_MCI = False):
    df_tabular = pd.DataFrame(
        columns=['ID', 'ses', 'Ventricles', 'Hippocampus', 'Entorhinal', 'Fusiform', 'MidTemp', 'WholeBrain', 'age',
                 'education', 'label'])

    for subject in list_ids:
        rows_df = adni_merged.loc[adni_merged['RID'] == subject]
        # sort rows by examination date
        rows_df['EXAMDATE'] = pd.to_datetime(rows_df['EXAMDATE'], format='%d/%m/%Y')
        rows_df.sort_values('EXAMDATE', inplace=True)
        i = 0

        while i < rows_df.shape[0] - 1:
            label = rows_df['DX'].iloc[i]
            if MCI or label != 'MCI' or extract_MCI:
                current_ses = rows_df['Month_bl'].iloc[i]
                next_ses = rows_df['Month_bl'].iloc[i + 1]

                if not extract_MCI or label == 'MCI':
                    new_row = create_row(rows_df=rows_df, i=i, subject=subject)
                    df_tabular = df_tabular.append(new_row, ignore_index=True)

                # if next session is within 5.8 months, it is considered 1 examination -> skip results from next session
                if (next_ses - current_ses) < 5.8:
                    i += 1
            i += 1

        if i < rows_df.shape[0] and (rows_df['DX'].iloc[i] != 'MCI' or MCI) and (not extract_MCI or rows_df['DX'].iloc[i] == 'MCI'):
            new_row = create_row(rows_df=rows_df, i=i, subject=subject)
            df_tabular = df_tabular.append(new_row, ignore_index=True)

    return df_tabular

def create_row(rows_df, i, subject):
    ventr = rows_df['Ventricles'].iloc[i]
    hippo = rows_df['Hippocampus'].iloc[i]
    wb = rows_df['WholeBrain'].iloc[i]
    ento = rows_df['Entorhinal'].iloc[i]
    fusi = rows_df['Fusiform'].iloc[i]
    midTemp = rows_df['MidTemp'].iloc[i]
    icv = rows_df['ICV'].iloc[i]
    age = rows_df['AGE'].iloc[i] + rows_df['Month_bl'].iloc[i]*(1/12)
    label = rows_df['DX'].iloc[i]
    education = rows_df['PTEDUCAT'].iloc[i]
    current_ses = rows_df['EXAMDATE'].iloc[i]

    new_row = {'ID': subject, 'ses': current_ses, 'Ventricles': ventr / icv,
               'Hippocampus': hippo / icv,
               'Entorhinal': ento / icv, 'Fusiform': fusi / icv, 'MidTemp': midTemp / icv, 'WholeBrain': wb / icv,
               'age': age, 'education': education, 'label': label}

    return new_row

def extractMCI_val():
    with open('../data_split_tabular_bio.json', 'r') as f:
        dict_split = json.load(f)

    adni_merged = pd.read_csv('../../Adni_merged.csv').dropna(subset = ['EXAMDATE', 'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp', 'ICV', 'AGE', 'PTEDUCAT', 'DX'])
    df_tabular = create_csv(adni_merged = adni_merged, list_ids = dict_split['val'], MCI = True, extract_MCI = True)

    path_save = os.path.join(os.getcwd(), f'./val_MCI_extracted_data_bio.csv')
    df_tabular.to_csv(path_save)

if __name__ == "__main__":
    writeTables(MCI=False)

    #extractMCI_val()

    print('CSV files for each set was created!')