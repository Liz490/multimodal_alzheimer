#!/usr/bin/env python3

from pathlib import Path
from pkg.models.pet_models.pet_cnn import Small_PET_CNN
from pkg.utils.dataloader import MultiModalDataset
from pkg.utils.test import test

from pkg.utils.load_path_config import load_path_config


def pet_testset(hparams: dict,
                test_csv_path: Path):
    normalization_pet = {'mean': hparams['norm_mean'],
                         'std': hparams['norm_std']}

    testset = MultiModalDataset(path=test_csv_path,
                                modalities=['pet1451'],
                                normalize_pet=normalization_pet,
                                binary_classification=hparams['n_classes'])

    return testset


def pet_model(checkpoint_path: Path):
    model = Small_PET_CNN.load_from_checkpoint(checkpoint_path)
    return model


if __name__ == '__main__':
    paths = load_path_config()

    # Two class
    model = pet_model(paths["pet_cnn_2_class"])
    testset = pet_testset(model.hparams, paths["test_set_csv"])
    experiment_name = 'test_set_pet_cnn_2_class'
    test(testset, model, experiment_name)

    # Three class
    model = pet_model(paths["pet_cnn_3_class"])
    testset = pet_testset(model.hparams, paths["test_set_csv"])
    experiment_name = 'test_set_pet_cnn_3_class'
    test(testset, model, experiment_name)
