from pkg.models.fusion_models.all_modalities_fusion import All_Modalities_Fusion
from pkg.models.mri_models.anat_cnn import Anat_CNN
from pkg.models.pet_models.pet_cnn import Small_PET_CNN
from pkg.utils.dataloader import MultiModalDataset
from pkg.utils.test import test
from pkg.utils.load_path_config import load_path_config


def all_mod_testset(paths: dict):
    model_pet = Small_PET_CNN.load_from_checkpoint(paths['pet_cnn_2_class'])
    model_mri = Anat_CNN.load_from_checkpoint(paths['mri_cnn_2_class'])
    testset = MultiModalDataset(
        path=paths['test_set_csv'],
        modalities=['pet1451', 't1w', 'tabular'],
        normalize_mri={'per_scan_norm': 'min_max'},
        normalize_pet={'mean': model_pet.hparams['norm_mean'],
                       'std': model_pet.hparams['norm_std']},
        binary_classification=model_pet.hparams['n_classes'],
        quantile=model_mri.hparams['norm_percentile'])
    return testset


def all_mod_model(paths: dict):
    model = All_Modalities_Fusion.load_from_checkpoint(
        paths['all_mod_2_class']
    )
    return model


if __name__ == '__main__':
    paths = load_path_config()
    testset = all_mod_testset(paths)
    model = all_mod_model(paths)
    experiment_name = 'test_set_' + paths['all_mod_2_class'].parents[1].name
    test(testset, model, experiment_name)
