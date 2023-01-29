from pathlib import Path
from pkg.models.pet_models.pet_cnn import Small_PET_CNN
from pkg.utils.dataloader import MultiModalDataset
from pkg.utils.test import test

LOG_DIRECTORY = 'lightning_logs'

PATH_PET_CNN_2_CLASS = Path('/data2/practical-wise2223/adni/adni_1/lightning_checkpoints/lightning_logs/best_runs/pet_2_class/checkpoints/epoch=112-step=112.ckpt')
HPARAMS_PET_CNN_2_CLASS = {
    'batch_size': 64,
    'norm_mean': 0.5145,
    'norm_std': 0.5383,
    'n_classes': 2
}


def pet_dataset_and_model(binary_classification: bool,
                          hparams: dict,
                          test_csv_path: Path,
                          checkpoint_path: Path):
    normalization_pet = {'mean': hparams['norm_mean'],
                         'std': hparams['norm_std']}

    testset = MultiModalDataset(path=test_csv_path,
                                modalities=['pet1451'],
                                normalize_pet=normalization_pet,
                                binary_classification=binary_classification)

    model = Small_PET_CNN.load_from_checkpoint(checkpoint_path)

    return testset, model


if __name__ == '__main__':
    test(pet_dataset_and_model, PATH_PET_CNN_2_CLASS, HPARAMS_PET_CNN_2_CLASS)
