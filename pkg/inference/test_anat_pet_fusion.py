from pkg.models.fusion_models.anat_pet_fusion import Anat_PET_CNN
from pkg.models.mri_models.anat_cnn import Anat_CNN
from pkg.models.pet_models.pet_cnn import Small_PET_CNN
from pkg.utils.dataloader import MultiModalDataset
from pkg.utils.test import test
from pkg.utils.load_path_config import load_path_config


def anat_pet_testset(paths: dict):
    model_pet = Small_PET_CNN.load_from_checkpoint(paths['pet_cnn_2_class'])
    model_mri = Anat_CNN.load_from_checkpoint(paths['mri_cnn_2_class'])
    testset = MultiModalDataset(
        path=paths['test_set_csv'],
        modalities=['pet1451', 't1w'],
        normalize_mri={'per_scan_norm': 'min_max'},
        normalize_pet={
            'mean': model_pet.hparams['norm_mean'],
            'std': model_pet.hparams['norm_std']},
        binary_classification=model_mri.hparams['n_classes'],
        quantile=model_mri.hparams['norm_percentile'])
    return testset


def anat_pet_model(paths: dict):
    model = Anat_PET_CNN.load_from_checkpoint(
        paths['pet_mri_2_class'],
        path_pet=paths['pet_cnn_2_class'],
        path_mri=paths['mri_cnn_2_class']
    )
    return model


if __name__ == '__main__':
    paths = load_path_config()
    testset = anat_pet_testset(paths)
    model = anat_pet_model(paths)
    experiment_name = 'test_set_' + paths['pet_mri_2_class'].parents[1].name
    test(testset, model, experiment_name)