import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def train_decision_tree():
    data = get_data()
    X_train, y_train = data[0], data[1]
    X_test, y_test = data[2], data[3]


    tree_model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=1)
    clf = tree_model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


def get_data():
    test_data = pd.read_csv('test_tabular_data.csv', sep=',', header=None).to_numpy()
    train_data = pd.read_csv('train_tabular_data.csv', sep=',', header=None).to_numpy()

    X_train = np.delete(train_data, [0,1,2,-1], 1)
    X_train = np.delete(X_train, (0), axis=0)
    y_train = train_data[:,-1]
    y_train = np.delete(y_train, (0), axis=0)
    X_test = np.delete(test_data, [0,1,2,-1], 1)
    X_test = np.delete(X_test, (0), axis=0)
    y_test = test_data[:,-1]
    y_test = np.delete(y_test, (0), axis=0)

    encoded_labels = encode_labels(y_train, y_test)
    y_test = encoded_labels[0]
    y_train = encoded_labels[1]


    return X_train, y_train, X_test, y_test

def encode_labels(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    print(le.classes_)
    y_test = le.transform(y_test)
    y_train = le.transform(y_train)
    return y_test, y_train


if __name__ == "__main__":
    train_decision_tree()
