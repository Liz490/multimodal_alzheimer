from pkg.models.fusion_models.pet_tabular_fusion import PET_TABULAR_CNN
from pkg.models.pet_models.pet_cnn import Small_PET_CNN
from pkg.utils.dataloader import MultiModalDataset
from pkg.utils.test import test
from pkg.utils.load_path_config import load_path_config


def pet_tab_testset(paths: dict):
    model_pet = Small_PET_CNN.load_from_checkpoint(paths['pet_cnn_2_class'])
    testset = MultiModalDataset(
        path=paths['test_set_csv'],
        modalities=['pet1451', 'tabular'],
        normalize_pet={
            'mean': model_pet.hparams['norm_mean'],
            'std': model_pet.hparams['norm_std']},
        binary_classification=model_pet.hparams['n_classes'])
    return testset


def pet_tab_model(paths: dict):
    model = PET_TABULAR_CNN.load_from_checkpoint(
        paths['pet_tab_2_class'],
        path_pet=paths['pet_cnn_2_class']
    )
    return model


if __name__ == '__main__':
    paths = load_path_config()
    testset = pet_tab_testset(paths)
    model = pet_tab_model(paths)
    experiment_name = 'test_set_' + paths['pet_tab_2_class'].parents[1].name
    test(testset, model, experiment_name)
