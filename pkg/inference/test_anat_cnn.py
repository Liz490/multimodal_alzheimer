from pathlib import Path
from pkg.models.mri_models.anat_cnn import Anat_CNN
from pkg.utils.dataloader import MultiModalDataset
from pkg.utils.test import test

LOG_DIRECTORY = 'lightning_logs'

PATH_ANAT_CNN_2_CLASS = Path('/data2/practical-wise2223/adni/adni_1/lightning_checkpoints/lightning_logs/best_runs/mri_2_class/checkpoints/epoch=37-step=37.ckpt')
HPARAMS_ANAT_CNN_2_CLASS = {
    'batch_size': 64,
    "norm_percentile": 0.98,
    'n_classes': 2
}


def anat_testset_and_model(binary_classification: bool,
                           hparams: dict,
                           test_csv_path: Path,
                           checkpoint_path: Path):

    testset = MultiModalDataset(path=test_csv_path,
                                modalities=['t1w'],
                                normalize_mri={'per_scan_norm': 'min_max'},
                                binary_classification=binary_classification,
                                quantile=hparams['norm_percentile'])

    model = Anat_CNN.load_from_checkpoint(checkpoint_path)

    return testset, model


if __name__ == '__main__':
    test(anat_testset_and_model,
         PATH_ANAT_CNN_2_CLASS,
         HPARAMS_ANAT_CNN_2_CLASS)
