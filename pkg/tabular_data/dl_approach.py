"""Fits a DL-classifier for the tabular adni data

See "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"
https://arxiv.org/abs/2207.01848
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import tabpfn
from pkg.tabular_data.data_preparation import *
import torch
from pkg.dataloader import MultiModalDataset
from torch.utils.data import DataLoader

def train_and_predict(val_path, train_path, storage_path, binary_classification):
    """Trains the TabPFN classifier for tabular data
            Args:
                val_data_path: path to file containins validation data
                train_data_path: path to file containins training data
    """

    x_train, y_train = get_data(train_path, binary_classification=binary_classification)
    print(f'shape input data: {x_train.shape}')
    classifier = train(x_train, y_train)

    x_val, y_val = get_data(val_path, binary_classification=binary_classification)
    y_eval, p_eval = classifier.predict(x_val, return_winning_probability=True)

    y_eval_train = classifier.predict(x_train, return_winning_probability=False)

    # Display metrics
    metrics.ConfusionMatrixDisplay.from_predictions(y_val, y_eval, cmap='Blues', colorbar=False,
                                                    display_labels=('NC_val', 'AD_val'))

    metrics.ConfusionMatrixDisplay.from_predictions(y_val, y_eval, cmap='Blues', colorbar=False,
                                                    display_labels=('NC_train', 'AD_train'))
    plt.show()
    f1_score_val = metrics.f1_score(y_val, y_eval)
    f1_score_train = metrics.f1_score(y_train, y_eval_train)
    print(f"validation F1-score: {f1_score_val}")
    print(f"training F1-score: {f1_score_train}")
    activation = None

    # Save model
    torch.save({'model_state_dict': classifier.model[2].state_dict(), 'tabular_baseline_F1':f1_score_train}, storage_path)
    return classifier

def train(x_train, y_train):

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
    x_train, y_train = get_data(path, binary_classification=binary_classification)
    classifier = train(x_train, y_train)
    return classifier


def get_avg_activation(activations):
    output = None
    for i in range(4):
        activ_ = activations[:, i:i + 1, :]
        output = activ_ if output is None else output + activ_
    output = output / 4
    output = torch.transpose(output, 0, 1).squeeze()
    return output

if __name__ == '__main__':
    VAL_PATH = '/vol/chameleon/projects/adni/adni_1/val_path_data_labels.csv'
    TRAIN_PATH = '/vol/chameleon/projects/adni/adni_1/train_path_data_labels.csv'
    STORAGE_PATH = '/vol/chameleon/projects/adni/adni_1/trained_models/tabular_baseline.pth'
    # Example usage how to extract probabilities of TabPFN
    # load classifier
    #classifier = load_model(VAL_PATH, TRAIN_PATH, STORAGE_PATH, True)
    classifier = train_and_predict(VAL_PATH, TRAIN_PATH, STORAGE_PATH, True)

    X_val, Y_val = get_data(VAL_PATH, True)
    X_train, Y_train = get_data(TRAIN_PATH, True)

    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
            handle.remove()
        return hook

    def get_avg_activation(activations):
        # print(activations)
        # activations = torch.cat(activations, 1)
        output = None
        for i in range(4):
            activ_ = activations[:, i:i + 1, :]
            output = activ_ if output is None else output + activ_
        output = output / 4
        output = torch.transpose(output, 0, 1).squeeze()
        return output

    handle = classifier.model[2].decoder[0].register_forward_hook(get_features('dec'))

    output = classifier.predict_proba(X_val, normalize_with_test=False)
    name = "dec"
    print(f'these are the format fetures asdflkajsöldfjkasdölfj: {features[name].shape}')
    activations = features[name]
    activations = get_avg_activation(activations)
    print(activations.shape)
    print(X_train.shape)

