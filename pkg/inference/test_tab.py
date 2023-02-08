#!/usr/bin/env python3

import torch

from pkg.models.tabular_models.tabular_pl_wrapper import Tabular_Model
from pkg.utils.dataloader import MultiModalDataset
from pkg.utils.test import test
from pkg.utils.load_path_config import load_path_config


def tab_testset(paths: dict, binary_classification: bool = True):
    testset = MultiModalDataset(
        path=paths['test_set_csv'],
        modalities=['pet1451', 't1w', 'tabular'],
        binary_classification=binary_classification)
    return testset


def tab_model(binary_classification: bool = True):
    if binary_classification:
        hparams = {
            'n_classes': 2,
            'ensemble_size': 4,
            'batch_size': 64,
            'loss_class_weights': torch.tensor([
                0.20314960629921264,
                0.7968503937007874
            ], dtype=torch.double)
        }
        model = Tabular_Model(hparams)
    else:
        hparams = {
            'n_classes': 3,
            'ensemble_size': 4,
            'batch_size': 64,
            'loss_class_weights': torch.tensor([
                0.4651162790697675,
                0.6712473572938689,
                0.8636363636363636
            ], dtype=torch.double)
        }
        model = Tabular_Model(hparams)
    return model


if __name__ == '__main__':
    paths = load_path_config()

    # Two class
    testset = tab_testset(paths, binary_classification=True)
    model = tab_model(binary_classification=True)
    experiment_name = 'test_set_tab_2_class'
    test(testset, model, experiment_name)

    # Three class
    testset = tab_testset(paths, binary_classification=False)
    model = tab_model(binary_classification=False)
    experiment_name = 'test_set_tab_3_class'
    test(testset, model, experiment_name)
