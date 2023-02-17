"""Fits a DL-classifier for the tabular adni data

See "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second"
https://arxiv.org/abs/2207.01848
"""

import matplotlib.pyplot as plt
from sklearn import metrics
import tabpfn
from pkg.models.tabular_models.data_preparation import *
import torch
from pkg.utils.dataloader import MultiModalDataset
from torch.utils.data import DataLoader

def train_and_predict(val_path, train_path, storage_path, binary_classification, ensemble_size = 4):
    """Trains the TabPFN classifier for tabular data
            Args:
                val_data_path: path to file containins validation data
                train_data_path: path to file containins training data
    """

    x_train, y_train = get_data(train_path, binary_classification=binary_classification)
    classifier = train(x_train, y_train, ensemble_size= ensemble_size)

    x_val, y_val = get_data(val_path, binary_classification=binary_classification)
    y_eval, p_eval = classifier.predict(x_val, return_winning_probability=True)

    y_eval_train = classifier.predict(x_train, return_winning_probability=False)

    # Display metrics
    metrics.ConfusionMatrixDisplay.from_predictions(y_val, y_eval, cmap='Blues', colorbar=False,
                                                    display_labels=('NC_val', 'AD_val'))

    metrics.ConfusionMatrixDisplay.from_predictions(y_val, y_eval, cmap='Blues', colorbar=False,
                                                    display_labels=('NC_train', 'AD_train'))
    plt.show()

    f1_score_val = metrics.f1_score(y_val, y_eval, average='macro')
    f1_score_train = metrics.f1_score(y_train, y_eval_train, average='macro')
    print(f"validation F1-score: {f1_score_val}")
    print(f"training F1-score: {f1_score_train}")

    # Save model
    torch.save({'model_state_dict': classifier.model[2].state_dict(), 'tabular_baseline_F1':f1_score_train}, storage_path)
    return classifier

def train(x_train, y_train, ensemble_size):

    # N_ensemble_configurations defines how many estimators are averaged, it is bounded by #features * #classes,
    # more ensemble members are slower, but more accurate
    classifier = tabpfn.TabPFNClassifier(device='cuda', N_ensemble_configurations=ensemble_size)
    classifier.fit(x_train, y_train, overwrite_warning=True)

    return classifier

def predict_batch(batch, classifier):

    samples = batch.numpy()
    logits = classifier.predict_proba(samples, normalize_with_test=False)
    pred = np.argmax(logits, axis=-1)
    pred = classifier.classes_.take(np.asarray(pred, dtype=np.intp))
    return pred, logits


def load_model(path, binary_classification = True, ensemble_size = 4):
    x_train, y_train = get_data(path, binary_classification=binary_classification)
    classifier = train(x_train, y_train, ensemble_size)
    return classifier, x_train.shape[0]


def get_avg_activation(activations, num_ensemble, training_size):
    output = None
    for i in range(num_ensemble):
        activ_ = activations[training_size:, i:i + 1, :]
        output = activ_ if output is None else output + activ_
    output = output / num_ensemble
    output = torch.transpose(output, 0, 1).squeeze(dim=0)
    return output

def calculate_statistics(classifier, path, binary_classification):
    x_test, y_test = get_data(path, binary_classification=binary_classification)
    y_eval, p_eval = classifier.predict(x_test, return_winning_probability=True)
    f1_score_test = metrics.f1_score(y_test, y_eval, average='macro')

    print(f'f1 score: {f1_score_test}')


if __name__ == '__main__':
    VAL_PATH = '/vol/chameleon/users/schmiere/Documents/Code /adlm_adni/data/val_path_data_labels.csv'
    TRAIN_PATH = '/vol/chameleon/users/schmiere/Documents/Code /adlm_adni/data/train_path_data_labels.csv'
    STORAGE_PATH = '/vol/chameleon/projects/adni/adni_1/trained_models/tabular_baseline.pth'
    test_path = '/vol/chameleon/users/schmiere/Documents/Code /adlm_adni/data/test_path_data_labels.csv'

    classifier = train_and_predict(VAL_PATH, TRAIN_PATH, STORAGE_PATH, binary_classification=True, ensemble_size=4)
    classifier_3c = train_and_predict(VAL_PATH, TRAIN_PATH, STORAGE_PATH, binary_classification=False, ensemble_size=4)

    calculate_statistics(classifier, test_path, True)
    calculate_statistics(classifier_3c, test_path, False)

