import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# PLOT LABEL DISTRIBUTIONS FOR ALL 3 SPLITS
df_splits = pd.DataFrame()
trainpath = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')
valpath = os.path.join(os.getcwd(), 'data/val_path_data_labels.csv')
testpath = os.path.join(os.getcwd(), 'data/test_path_data_labels.csv')
df_train = pd.read_csv(trainpath)
df_train['mode'] = 'train'

df_val = pd.read_csv(valpath)
df_val['mode'] = 'val'
df_test = pd.read_csv(testpath)
df_test['mode'] = 'test'
df_all = pd.concat([df_train, df_val, df_test])

fig1, ax = plt.subplots(figsize=(12,6))
fig2, ax = plt.subplots(figsize=(12,6))

# for countplot uncomment the following 
blues = sns.color_palette("Blues", n_colors=3)
#dark_blues = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
sns.despine()
cntplt = sns.countplot(data=df_all, x="mode", hue="label", palette=['#b0cffb', '#7f96cf', '#22418e'])#, ax=ax[0])#palette=blues)  lightsteelblue ['royalblue', 'cornflowerblue' , 'lightsteelblue']

cntplt.set(xlabel ="", ylabel = "")

current_handles, current_labels = plt.gca().get_legend_handles_labels()
current_labels[2] = 'AD'

# # sort or reorder the labels and handles
# reversed_handles = list(reversed(current_handles))
# reversed_labels = list(reversed(current_labels))

# # call plt.legend() with the new values
plt.legend(current_handles, current_labels, frameon=False)
plt.gcf().subplots_adjust(bottom=0.15)
# plt.legend()
path_save1 = '/u/home/eisln/adlm_adni/plots_presentation/splits.png'
plt.savefig(path_save1, dpi=300, transparent=True)
plt.close()
# ax[0].legend([],[], frameon=False)
# cntplt.legend.remove()

# dist_train = df_train['label'].value_counts()
# df_splits = df_splits.append({'mode': 'train', 'CN': dist_train['CN'], 'MCI': dist_train['MCI'], 'Dementia': dist_train['Dementia']}, ignore_index=True)
# dist_val = df_val['label'].value_counts()
# df_splits = df_splits.append({'mode': 'val', 'CN': dist_val['CN'], 'MCI': dist_val['MCI'], 'Dementia': dist_val['Dementia']}, ignore_index=True)
# dist_test = df_test['label'].value_counts()
# df_splits = df_splits.append({'mode': 'test', 'CN': dist_test['CN'], 'MCI': dist_test['MCI'], 'Dementia': dist_test['Dementia']}, ignore_index=True)

# print(df_splits)
# splitsplot = df_splits.set_index('mode').plot(kind='bar', stacked=True, color=['lightsteelblue', 'cornflowerblue', 'royalblue'])



# plt.savefig('/u/home/eisln/adlm_adni/plots_presentation/distribution_train_val_test.png')
# plt.close()

# plt.xticks(rotation=45)

# for barplot uncomment the following
# sns.barplot(x = df_all["label"])
# plt.xlabel("labels")
# plt.savefig('/u/home/eisln/adlm_adni/plots_presentation/distribution_train_val_test.png')


# PLOT LABEL DISTRIBUTION FOR ALL 7 MODELS IN THE 3 STAGES OF TRAINSET
from pkg.utils.dataloader import MultiModalDataset

trainpath = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')

normalization_pet = {'mean': 0.5145, 'std': 0.5383}
normalization_mri = {'per_scan_norm': 'min_max'}
df_stages = pd.DataFrame()
# stage 1 
trainset_pet = MultiModalDataset(path=trainpath, modalities=['pet1451'])
trainset_mri = MultiModalDataset(path=trainpath, modalities=['t1w'])
trainset_tabular = MultiModalDataset(path=trainpath, modalities=['tabular'])
counts_pet, _ = trainset_pet.get_label_distribution()
counts_pet = counts_pet.numpy()
counts_mri, _ = trainset_mri.get_label_distribution()
counts_mri = counts_mri.numpy()
counts_tabular, _ = trainset_tabular.get_label_distribution()
counts_tabular = counts_tabular.numpy()

df_stages = df_stages.append({'mode': 'TAB', 'CN': counts_tabular[0], 'MCI': counts_tabular[1], 'Dementia': counts_tabular[2]},
                            ignore_index=True)

df_stages = df_stages.append({'mode': 'MRI', 'CN': counts_mri[0], 'MCI': counts_mri[1], 'Dementia': counts_mri[2]},
                            ignore_index=True)

df_stages = df_stages.append({'mode': 'PET', 'CN': counts_pet[0], 'MCI': counts_pet[1], 'Dementia': counts_pet[2]},
                            ignore_index=True)
# stage 2 
trainset_pet_mri = MultiModalDataset(path=trainpath, modalities=['pet1451', 't1w'])
trainset_mri_tab = MultiModalDataset(path=trainpath, modalities=['t1w', 'tabular'])
trainset_pet_tabular = MultiModalDataset(path=trainpath, modalities=['pet1451', 'tabular'])
counts_pet_mri, _ = trainset_pet_mri.get_label_distribution()
counts_pet_mri = counts_pet_mri.numpy()
counts_mri_tab, _ = trainset_mri_tab.get_label_distribution()
counts_mri_tab = counts_mri_tab.numpy()
counts_pet_tabular, _ = trainset_pet_tabular.get_label_distribution()
counts_pet_tabular = counts_pet_tabular.numpy()

df_stages = df_stages.append({'mode': 'PET-MRI', 'CN': counts_pet_mri[0], 'MCI': counts_pet_mri[1], 'Dementia': counts_pet_mri[2]},
                            ignore_index=True)

df_stages = df_stages.append({'mode': 'MRI-TAB', 'CN': counts_mri_tab[0], 'MCI': counts_mri_tab[1], 'Dementia': counts_mri_tab[2]},
                            ignore_index=True)

df_stages = df_stages.append({'mode': 'PET-TAB', 'CN': counts_pet_tabular[0], 'MCI': counts_pet_tabular[1], 'Dementia': counts_pet_tabular[2]},
                            ignore_index=True)
# stage 3

trainset_all = MultiModalDataset(path=trainpath, modalities=['pet1451', 't1w', 'tabular'])
counts_all, _ = trainset_all.get_label_distribution()
counts_all = counts_all.numpy()

df_stages = df_stages.append({'mode': 'ALL', 'CN': counts_all[0], 'MCI': counts_all[1], 'Dementia': counts_all[2]},
                            ignore_index=True)

print(df_stages)
sns.despine()
stagesplot = df_stages.set_index('mode').plot(kind='bar', stacked=True, rot=45, color=['#b0cffb', '#7f96cf', '#22418e']) #, ax=ax[1]
stagesplot.set(xlabel ="", ylabel = "")
sns.despine()
plt.gcf().subplots_adjust(bottom=0.15)
current_handles2, current_labels2 = plt.gca().get_legend_handles_labels()
current_labels2[2] = 'AD'
# # sort or reorder the labels and handles
# reversed_handles = list(reversed(current_handles))
# reversed_labels = list(reversed(current_labels))

# # call plt.legend() with the new values
plt.legend(current_handles2, current_labels2, frameon=False)
plt.gcf().subplots_adjust(bottom=0.15)
# stagesplot.set(xlabel ="", ylabel = "")
#ax[1].plot(df_stages.set_index('mode'), kind='bar', stacked=True)#, color=['royalblue', 'cornflowerblue' , 'lightsteelblue'])

path_save2 = '/u/home/eisln/adlm_adni/plots_presentation/stages.png'
plt.savefig(path_save2, dpi=300, transparent=True)