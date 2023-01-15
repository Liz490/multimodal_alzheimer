"""Fits a DL-classifier for the tabular adni data

See "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"
https://arxiv.org/abs/2207.01848
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import tabpfn
from tabular_data import data_preparation
import torch
from pkg.dataloader import MultiModalDataset
from torch.utils.data import DataLoader

def train_and_predict(val_data_path, train_data_path, storage_path, binary_classification):
    """Trains the TabPFN classifier for tabular data
            Args:
                val_data_path: path to file containins validation data
                train_data_path: path to file containins training data
    """

    classifier = train(train_data_path, binary_classification)

    x_val, y_val = data_preparation.get_data(val_data_path, binary_classification)
    y_eval, p_eval = classifier.predict(x_val, return_winning_probability=True)

    # Display metrics
    metrics.ConfusionMatrixDisplay.from_predictions(y_val, y_eval, cmap='Blues', colorbar=False,
                                                    display_labels=('NC', 'AD'))
    plt.show()
    f1_score = metrics.f1_score(y_val, y_eval)
    print(f"F1-score: {f1_score}")

    # Save model
    torch.save({'model_state_dict': classifier.model[2].state_dict(), 'tabular_baseline_F1':f1_score}, storage_path)
    return classifier

def train(x_train, y_train, binary_classification):
    #x_train, y_train = data_preparation.get_data(train_data_path, binary_classification)

    # N_ensemble_configurations defines how many estimators are averaged, it is bounded by #features * #classes,
    # more ensemble members are slower, but more accurate
    classifier = tabpfn.TabPFNClassifier(device='cuda', N_ensemble_configurations=6)
    classifier.fit(x_train, y_train, overwrite_warning=True)

    return classifier

def predict_batch(batch, classifier):

    samples = batch.numpy()
    logits = classifier.predict_proba(samples, normalize_with_test=False)
    pred = np.argmax(logits, axis=-1)
    pred = classifier.classes_.take(np.asarray(pred, dtype=np.intp))
    return pred, logits


def load_model(path, binary_classification = True):
    """
    loads model from directory
        Args:
            path: path to location where model is stored
        Returns:
            TabPFNlassifier with stored weights
    """
    VAL_PATH = '/vol/chameleon/projects/adni/adni_1/val_path_data_labels.csv'
    TRAIN_PATH = '/vol/chameleon/projects/adni/adni_1/train_path_data_labels.csv'
    """def load():
        classifier = tabpfn.TabPFNClassifier(device='cuda', N_ensemble_configurations=4)
        checkpoint = torch.load(path)
        classifier.model[2].load_state_dict(checkpoint['model_state_dict'])
        return classifier
    try:
        classifier = load()
    except:
    print('No pre-trained model available. \n Train model')
    """
    classifier = train_and_predict(VAL_PATH, TRAIN_PATH, path, binary_classification)
    return classifier


if __name__ == '__main__':
    VAL_PATH = '/vol/chameleon/projects/adni/adni_1/val_path_data_labels.csv'
    TRAIN_PATH = '/vol/chameleon/projects/adni/adni_1/train_path_data_labels.csv'
    STORAGE_PATH = '/vol/chameleon/projects/adni/adni_1/trained_models/tabular_baseline.pth'
    train_and_predict(VAL_PATH, TRAIN_PATH, STORAGE_PATH, True)

    # Example usage how to extract probabilities of TabPFN
    # load classifier
    #classifier = load_model(VAL_PATH, TRAIN_PATH, STORAGE_PATH, True)
    classifier = train(TRAIN_PATH, True)

    valset = MultiModalDataset(path=VAL_PATH,
                               modalities=['tabular'],
                               binary_classification=True)

    valloader = DataLoader(
        valset,
        batch_size=5,
        shuffle=False)

    for batch in valloader:
        y, p = predict_batch(batch['tabular'], classifier)


