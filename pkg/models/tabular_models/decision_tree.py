"""Trains a decision tree for the tabular adni data"""
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import data_preparation
import os
from pkg.utils.dataloader import MultiModalDataset


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

if __name__ == "__main__":
    VAL_DATA_PATH = os.path.join(os.getcwd(), 'data/val_path_data_labels.csv')
    TRAIN_DATA_PATH = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')
    model = train_decision_tree(VAL_DATA_PATH, TRAIN_DATA_PATH, balanced='balanced')


