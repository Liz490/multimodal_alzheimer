import os
import json
import pandas as pd
from datetime import datetime


def writeTables():
    with open('../data_split_tabular.json', 'r') as f:
        dict_split = json.load(f)

    adni_merged = pd.read_csv('../../Adni_merged.csv').dropna()

    for mode in ['train', 'val', 'test']:
        list_ids = dict_split[mode]
        df_tabular = pd.DataFrame(columns=['ID', 'ses', 'cdrsb', 'ldeltotal', 'mpaccdigit', 'mpacctrailsb', 'mmse', 'label'])

        for subject in list_ids:
            rows_df = adni_merged.loc[adni_merged['index'] == subject]
            #sort rows by examination date
            rows_df['EXAMDATE'] = pd.to_datetime(rows_df['EXAMDATE'], format='%d/%m/%Y')
            rows_df.sort_values('EXAMDATE', inplace= True)
            i = 0
            while i < rows_df.shape[0]-1:
                current_ses = rows_df['EXAMDATE'].iloc[i]
                next_ses = rows_df['EXAMDATE'].iloc[i+1]

                #if next session is within 6 months, it is considered 1 examination -> skip results from next session
                if (next_ses.year - current_ses.year) * 12 + next_ses.month - current_ses.month < 6:
                    i += 1

                new_row = {'ID': subject, 'ses': current_ses, 'cdrsb': rows_df['CDRSB'].iloc[i] ,
                           'ldeltotal':rows_df['LDELTOTAL'].iloc[i], 'mpaccdigit': rows_df['mPACCdigit'].iloc[i],
                           'mpacctrailsb': rows_df['mPACCtrailsB'].iloc[i], 'mmse':  rows_df['MMSE'].iloc[i], 'label': rows_df['DX'].iloc[i]}

                df_tabular = df_tabular.append(new_row, ignore_index=True)
                i += 1

            if i < rows_df.shape[0]:
                new_row = {'ID': subject, 'ses': rows_df['EXAMDATE'].iloc[i], 'cdrsb': rows_df['CDRSB'].iloc[i],
                           'ldeltotal': rows_df['LDELTOTAL'].iloc[i], 'mpaccdigit': rows_df['mPACCdigit'].iloc[i],
                           'mpacctrailsb': rows_df['mPACCtrailsB'].iloc[i], 'mmse': rows_df['MMSE'].iloc[i],
                           'label': rows_df['DX'].iloc[i]}
                df_tabular = df_tabular.append(new_row, ignore_index=True)

        path_save = os.path.join(os.getcwd(), f'./{mode}_tabular_data.csv')
        df_tabular.to_csv(path_save)

if __name__ == "__main__":
    writeTables()
    print('CSV files for each set was created!')