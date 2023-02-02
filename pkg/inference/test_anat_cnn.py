from pathlib import Path
from pkg.models.mri_models.anat_cnn import Anat_CNN
from pkg.utils.dataloader import MultiModalDataset
from pkg.utils.test import test

from pkg.utils.load_path_config import load_path_config


def anat_testset(hparams: dict,
                 test_csv_path: Path):
    testset = MultiModalDataset(path=test_csv_path,
                                modalities=['t1w'],
                                normalize_mri={'per_scan_norm': 'min_max'},
                                binary_classification=hparams['n_classes'],
                                quantile=hparams['norm_percentile'])

    return testset


def anat_model(checkpoint_path: Path):
    model = Anat_CNN.load_from_checkpoint(checkpoint_path)
    return model


if __name__ == '__main__':
    paths = load_path_config()

    # Two class
    model = anat_model(paths["mri_cnn_2_class"])
    testset = anat_testset(model.hparams, paths["test_set_csv"])
    experiment_name = 'test_set_mri_cnn_2_class'
    test(testset, model, experiment_name)

    # Three class
    model = anat_model(paths["mri_cnn_3_class"])
    testset = anat_testset(model.hparams, paths["test_set_csv"])
    experiment_name = 'test_set_mri_cnn_3_class'
    test(testset, model, experiment_name)
