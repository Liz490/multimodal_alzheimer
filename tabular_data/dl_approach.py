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
from pkg.dataloader import MultiModalDataset


def train(val_data_path, train_data_path, storage_path):
    """Trains the TabPFN classifier for tabular data
            Args:
                val_data_path: path to file containins validation data
                train_data_path: path to file containins training data
    """
    trainset_tabular = MultiModalDataset(path=train_data_path, binary_classification=True, modalities=['tabular'])
    data_train = trainset_tabular.df_tab
    valset_tabular = MultiModalDataset(path=val_data_path, binary_classification=True, modalities=['tabular'])
    data_val = valset_tabular.df_tab
    data = data_preparation.get_data(data_val, data_train)
    x_train, y_train = data[0], data[1]
    x_val, y_val = data[2], data[3]

    # N_ensemble_configurations defines how many estimators are averaged,
    # it is bounded by #features * #classes,
    # more ensemble members are slower, but more accurate
    classifier = tabpfn.TabPFNClassifier(device='cuda', N_ensemble_configurations=4)
    classifier.fit(x_train, y_train, overwrite_warning=True)
    y_eval, p_eval = classifier.predict(x_val, return_winning_probability=True)

    # Display metrics
    metrics.ConfusionMatrixDisplay.from_predictions(y_val, y_eval, cmap='Blues', colorbar=False,
                                                    display_labels=('NC', 'AD'))
    plt.show()
    f1_score = metrics.f1_score(y_val, y_eval)
    print(f"F1-score: {f1_score}")

    # Save model
    breakpoint()
    torch.save({'model_state_dict': classifier.model[2].state_dict(), 'tabular_baseline_F1':f1_score}, storage_path)


def load_model(path):
    """
    loads model from directory
        Args:
            path: path to location where model is stored
        Returns:
            TabPFNlassifier with stored weights
    """
    VAL_PATH = '/vol/chameleon/projects/adni/adni_1/val_path_data_labels.csv'
    TRAIN_PATH = '/vol/chameleon/projects/adni/adni_1/train_path_data_labels.csv'
    def load():
        classifier = tabpfn.TabPFNClassifier(device='cuda', N_ensemble_configurations=4)
        checkpoint = torch.load(path)
        classifier.model[2].load_state_dict(checkpoint['model_state_dict'])
        return classifier
    try:
        classifier = load()
    except:
        print('No pre-trained model available. \n Train model')
        train(VAL_PATH, TRAIN_PATH, path)
        classifier = load()
    return classifier


if __name__ == '__main__':
    VAL_PATH = '/vol/chameleon/projects/adni/adni_1/val_path_data_labels.csv'
    TRAIN_PATH = '/vol/chameleon/projects/adni/adni_1/train_path_data_labels.csv'
    STORAGE_PATH = '/vol/chameleon/projects/adni/adni_1/trained_models/tabular_baseline.pth'
    train(VAL_PATH, TRAIN_PATH, STORAGE_PATH)

