from torch.utils.data import DataLoader
from pathlib import Path
import pytorch_lightning as pl
from typing import Callable


def test(init_dataset_and_model: Callable,
         checkpoint_path: Path,
         hparams: dict):
    pl.seed_everything(5, workers=True)
    experiment_name = 'test_set_' + checkpoint_path.parents[1].name

    assert hparams['n_classes'] == 2 or hparams['n_classes'] == 3
    if hparams['n_classes'] == 2:
        binary_classification = True
    else:
        binary_classification = False

    test_csv_path = Path.cwd() / 'data/test_path_data_labels.csv'

    testset, model = init_dataset_and_model(binary_classification,
                                            hparams,
                                            test_csv_path,
                                            checkpoint_path)

    testloader = DataLoader(
        testset,
        batch_size=hparams['batch_size'],
        num_workers=32)

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir='lightning_logs',
        name=experiment_name)

    trainer = pl.Trainer(
        logger=tb_logger,
        log_every_n_steps=5,
        accelerator='gpu',
        devices=1
    )

    trainer.test(model, testloader)
