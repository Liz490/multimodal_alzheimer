import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def train_decision_tree():
    data = get_data()
    X_train, y_train = data[0], data[1]
    X_val, y_val = data[2], data[3]

    tree_model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=1, class_weight='balanced')
    clf = tree_model.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    print(f"F1 score: {metrics.f1_score(y_val, y_pred)}")
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_val, y_pred)
    plt.show(cmap = plt.cm.get_cmap('Blues'))

    print("Accuracy:", metrics.accuracy_score(y_val, y_pred))

def get_data():
    print(os.getcwd())
    val_data = pd.read_csv('val_tabular_data_bio.csv', sep=',', header=None).to_numpy()
    train_data = pd.read_csv('train_tabular_data_bio.csv', sep=',', header=None).to_numpy()

    X_train = np.delete(train_data, [0,1,2,-1], 1)
    X_train = np.delete(X_train, (0), axis=0)
    y_train = train_data[:,-1]
    y_train = np.delete(y_train, (0), axis=0)
    X_val = np.delete(val_data, [0,1,2,-1], 1)
    X_val = np.delete(X_val, (0), axis=0)
    y_val = val_data[:,-1]
    y_val = np.delete(y_val, (0), axis=0)

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

    encoded_labels = encode_labels(y_train, y_val)
    y_val = encoded_labels[0]
    y_train = encoded_labels[1]

    return X_train, y_train, X_val, y_val

def encode_labels(y_train, y_val):
    le = LabelEncoder()
    le.fit(y_train)
    #print(le.classes_)
    y_val = le.transform(y_val)
    y_train = le.transform(y_train)
    return y_val, y_train


if __name__ == "__main__":
    train_decision_tree()
