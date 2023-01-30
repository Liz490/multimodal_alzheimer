from pkg.models.fusion_models.tabular_mri_fusion import Tabular_MRT_Model
from pkg.models.mri_models.anat_cnn import Anat_CNN
from pkg.utils.dataloader import MultiModalDataset
from pkg.utils.test import test
from pkg.utils.load_path_config import load_path_config


def mri_tab_testset(paths: dict):
    model_mri = Anat_CNN.load_from_checkpoint(paths['mri_cnn_2_class'])
    testset = MultiModalDataset(
        path=paths['test_set_csv'],
        modalities=['t1w', 'tabular'],
        normalize_mri={'per_scan_norm': 'min_max'},
        binary_classification=model_mri.hparams['n_classes'],
        quantile=model_mri.hparams['norm_percentile'])
    return testset


def mri_tab_model(paths: dict):
    model = Tabular_MRT_Model.load_from_checkpoint(
        paths['mri_tab_2_class'],
        path_mri=paths['mri_cnn_2_class']
    )
    return model


if __name__ == '__main__':
    paths = load_path_config()
    testset = mri_tab_testset(paths)
    model = mri_tab_model(paths)
    experiment_name = 'test_set_' + paths['mri_tab_2_class'].parents[1].name
    test(testset, model, experiment_name)
