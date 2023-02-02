#!/usr/bin/env python3

from pkg.models.fusion_models.all_modalities_fusion import All_Modalities_Fusion
from pkg.models.mri_models.anat_cnn import Anat_CNN
from pkg.models.pet_models.pet_cnn import Small_PET_CNN
from pkg.utils.dataloader import MultiModalDataset
from pkg.utils.test import test
from pkg.utils.load_path_config import load_path_config


def all_mod_testset(paths: dict, binary_classification: bool = True):
    if binary_classification:
        model_pet = Small_PET_CNN.load_from_checkpoint(paths['pet_cnn_2_class'])
        model_mri = Anat_CNN.load_from_checkpoint(paths['mri_cnn_2_class'])
    else:
        model_pet = Small_PET_CNN.load_from_checkpoint(paths['pet_cnn_3_class'])
        model_mri = Anat_CNN.load_from_checkpoint(paths['mri_cnn_3_class'])
    testset = MultiModalDataset(
        path=paths['test_set_csv'],
        modalities=['pet1451', 't1w', 'tabular'],
        normalize_mri={'per_scan_norm': 'min_max'},
        normalize_pet={'mean': model_pet.hparams['norm_mean'],
                       'std': model_pet.hparams['norm_std']},
        binary_classification=model_pet.hparams['n_classes'],
        quantile=model_mri.hparams['norm_percentile'])
    return testset


def all_mod_model(paths: dict, binary_classification: bool = True):
    if binary_classification:
        model = All_Modalities_Fusion.load_from_checkpoint(
            paths['all_mod_2_class']
        )
    else:
        model = All_Modalities_Fusion.load_from_checkpoint(
            paths['all_mod_3_class']
        )
    return model


if __name__ == '__main__':
    paths = load_path_config()

    # Two class
    testset = all_mod_testset(paths, binary_classification=True)
    model = all_mod_model(paths, binary_classification=True)
    experiment_name = 'test_set_all_mod_2_class'
    test(testset, model, experiment_name)

    # Three class
    testset = all_mod_testset(paths, binary_classification=False)
    model = all_mod_model(paths, binary_classification=False)
    experiment_name = 'test_set_all_mod_3_class'
    test(testset, model, experiment_name)
