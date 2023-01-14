"""Fits a DL-classifier for the tabular adni data

See "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"
https://arxiv.org/abs/2207.01848
"""

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import tabpfn
import data_preparation
import torch


def train(val_data_path, train_data_path, storage_path):
    """Trains the TabPFN classifier for tabular data
            Args:
                val_data_path: path to file containins validation data
                train_data_path: path to file containins training data
    """
    data = data_preparation.get_data(val_data_path, train_data_path)
    x_train, y_train = data[0], data[1]
    x_val, y_val = data[2], data[3]

    # Encode labels
    lab_enc = LabelEncoder()
    lab_enc.fit(y_train)
    y_val = lab_enc.transform(y_val)
    y_train = lab_enc.transform(y_train)

    # N_ensemble_configurations defines how many estimators are averaged,
    # it is bounded by #features * #classes,
    # more ensemble members are slower, but more accurate
    classifier = TabPFNClassifier(device='cuda', N_ensemble_configurations=4)
    classifier.fit(x_train, y_train, overwrite_warning=True)
    y_eval, p_eval = classifier.predict(x_val, return_winning_probability=True)

    # Display metrics
    metrics.ConfusionMatrixDisplay.from_predictions(y_val, y_eval, cmap='Blues', colorbar=False,
                                                    display_labels=('NC', 'AD'))
    plt.show()
    print(f"F1-score: {metrics.f1_score(y_val, y_eval)}")

    # Save model
    torch.save(classifier.model[2].state_dict(), storage_path)


def load_model(path):
    """
    loads model from directory
        Args:
            path: path to location where model is stored
        Returns:
            TabPFNlassifier with stored weights
    """
    VAL_PATH = 'val_tabular_data.csv'
    TRAIN_PATH = 'train_tabular_data.csv'
    def load():
        classifier = TabPFNClassifier(device='cuda', N_ensemble_configurations=4)
        classifier.model[2].load_state_dict(torch.load(path))
        return classifier
    try:
        classifier = load()
    except:
        print('No pre-trained model available. \n Train model')
        train(VAL_PATH, TRAIN_PATH, path)
        classifier = load()
    return classifier


if __name__ == '__main__':
    VAL_PATH = 'val_tabular_data.csv'
    TRAIN_PATH = 'train_tabular_data.csv'
    STORAGE_PATH = ''
    train(VAL_PATH, TRAIN_PATH, STORAGE_PATH)

