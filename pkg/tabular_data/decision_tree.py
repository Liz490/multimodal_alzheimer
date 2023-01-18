"""Trains a decision tree for the tabular adni data"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import data_preparation
import os
from pkg.dataloader import MultiModalDataset

def train_decision_tree(val_data_path, train_data_path, balanced='unbalanced'):
    """Trains a decision tree for tabular data

            Args:
                val_data_path: path to file contains validation data
                train_data_path: path to file contains training data
                balanced: indicates whether training is balanced or not
            Returns:
                Trained decision tree
    """
    x_train, y_train = data_preparation.get_data(train_data_path, True)
    x_val, y_val = data_preparation.get_data(val_data_path, True)

    tree_model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=1, class_weight=balanced)
    clf = tree_model.fit(x_train, y_train)
    y_pred = clf.predict(x_val)

    print(f"F1 score: {metrics.f1_score(y_val, y_pred, average='micro')}")
    metrics.ConfusionMatrixDisplay.from_predictions(y_val, y_pred, cmap='Blues', colorbar=False, display_labels=('NC', 'AD'))
    plt.show()

    print("Accuracy:", metrics.accuracy_score(y_val, y_pred))
    return clf


def predict_mci(path, model):
    """Predicts samples with mci labels to inspect
    whether they are predicted as ad or nc in case of 2-class DT

            Args:
                path: path to file that contains mci data
                model: trained decision tree
    """
    mci_data = pd.read_csv(path, sep=',', header=None).to_numpy()
    x_mci = np.delete(mci_data, [0, 1, 2, -1], 1)
    x_mci = np.delete(x_mci, 0, axis=0)

    y_predict = model.predict(x_mci)

    ad_label = np.sum(np.where(y_predict == 1, 1, 0))
    cn_label = np.sum(np.where(y_predict == 0, 1, 0))
    print(f'Share of MCI samples predicted AD: {ad_label/(ad_label+cn_label)}.\n'
          f'Share of MCI samples predicted CN: {cn_label/(ad_label+cn_label)} ')


def calculate_statistic(y_val, y_train):
    """calculate label distribution for validation and training data
        Args:
            y_val: labels for validation data
            y_train: labels for training data
    """
    cn_val = np.count_nonzero(y_val == 'CN')
    mci_val = np.count_nonzero(y_val == 'MCI')
    ad_val = np.count_nonzero(y_val == 'Dementia')

    cn_train = np.count_nonzero(y_train == 'CN')
    mci_train = np.count_nonzero(y_train == 'MCI')
    ad_train = np.count_nonzero(y_train == 'Dementia')

    total_train = y_train.shape[0]
    total_val = y_val.shape[0]

    share_cn_train = cn_train/total_train
    share_mci_train = mci_train/total_train
    share_ad_train = ad_train/total_train

    share_cn_val = cn_val / total_val
    share_mci_val = mci_val / total_val
    share_ad_val = ad_val / total_val

    print(f'class distribution val data: CN: {cn_val}, MCI:{mci_val}, AD:{ad_val}')
    print(f'class distribution train data: CN: {cn_train}, MCI: {mci_train}, AD: {ad_train}')
    print(f'class distribution: CN train: {share_cn_train} CN val: {share_cn_val}')
    print(f'class distribution: MCI train: {share_mci_train} MCI val: {share_mci_val}')
    print(f'class distribution: AD train: {share_ad_train} AD val: {share_ad_val}')


if __name__ == "__main__":
    VALPATH = '/vol/chameleon/users/schmiere/Documents/Code /adlm_adni/data/val_path_data_labels.csv'
    TRAINPATH = '/vol/chameleon/users/schmiere/Documents/Code /adlm_adni/data/train_path_data_labels.csv'
    model = train_decision_tree(VALPATH, TRAINPATH, balanced='balanced')


