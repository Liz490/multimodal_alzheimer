#!/usr/bin/env python3

from pathlib import Path
from pkg.models.fusion_models.early_fusion import PET_MRI_EF
from pkg.utils.dataloader import MultiModalDataset
from pkg.utils.test import test

from pkg.utils.load_path_config import load_path_config
import sys


def ef_testset(hparams: dict,
                test_csv_path: Path):

    
    normalization_mri = {'per_scan_norm': 'min_max'}

    normalization_pet = {'mean': hparams['norm_mean'],
                         'std': hparams['norm_std']}

    testset = MultiModalDataset(path=test_csv_path,
                                modalities=['pet1451', 't1w'],
                                normalize_mri=normalization_mri,
                                normalize_pet=normalization_pet,
                                binary_classification=hparams['n_classes'],
                                quantile=hparams['norm_percentile'])

    return testset


def ef_model(checkpoint_path: Path):
    model = PET_MRI_EF.load_from_checkpoint(checkpoint_path)
    return model


if __name__ == '__main__':
    paths = load_path_config()
    # Two class
    model = ef_model(paths["early_fusion_different_norm_2_class"])
    testset = ef_testset(model.hparams, paths["test_set_csv"])
    experiment_name = 'test_set_early_fusion_different_normalization_2_class'
    test(testset, model, experiment_name)

