#!/usr/bin/env python3

from pkg.models.fusion_models.tabular_mri_fusion import Tabular_MRT_Model
from pkg.models.mri_models.anat_cnn import Anat_CNN
from pkg.utils.dataloader import MultiModalDataset
from pkg.utils.test import test
from pkg.utils.load_path_config import load_path_config


def mri_tab_testset(paths: dict, binary_classification: bool = True):
    if binary_classification:
        model_mri = Anat_CNN.load_from_checkpoint(paths['mri_cnn_2_class'])
    else:
        model_mri = Anat_CNN.load_from_checkpoint(paths['mri_cnn_3_class'])
    testset = MultiModalDataset(
        path=paths['test_set_csv'],
        modalities=['pet1451', 't1w', 'tabular'],
        normalize_mri={'per_scan_norm': 'min_max'},
        binary_classification=model_mri.hparams['n_classes'],
        quantile=model_mri.hparams['norm_percentile'])
    return testset


def mri_tab_model(paths: dict, binary_classification: bool = True):
    if binary_classification:
        model = Tabular_MRT_Model.load_from_checkpoint(
            paths['mri_tab_2_class'],
            path_mri=paths['mri_cnn_2_class']
        )
    else:
        model = Tabular_MRT_Model.load_from_checkpoint(
            paths['mri_tab_3_class'],
            path_mri=paths['mri_cnn_3_class']
        )
    return model


if __name__ == '__main__':
    paths = load_path_config()
    
    # Two class
    testset = mri_tab_testset(paths, binary_classification=True)
    model = mri_tab_model(paths, binary_classification=True)
    experiment_name = 'test_set_mri_tab_2_class'
    test(testset, model, experiment_name)

    # Three class
    testset = mri_tab_testset(paths, binary_classification=False)
    model = mri_tab_model(paths, binary_classification=False)
    experiment_name = 'test_set_mri_tab_3_class'
    test(testset, model, experiment_name)
