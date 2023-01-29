from torch.utils.data import DataLoader
from pathlib import Path
import pytorch_lightning as pl
from typing import Callable


def test(init_dataset_and_model: Callable,
         checkpoint_path: Path,
         hparams: dict):
    """
    Test a model on the test set. Logs results to tensorboard under
    the name of the checkpoints parent directory. I.e.
    lightning_logs/test_set_<name of checkpoint parent directory>.

    Args:
        init_dataset_and_model (Callable): Function that returns a testset and
            a model. The function should take the following arguments:
                binary_classification (bool): Whether the model is a binary
                    classification model.
                hparams (dict): Hyperparameters of the models dataloader.
                test_csv_path (Path): Path to the csv file containing the test
                    set paths and labels.
                checkpoint_path (Path): Path to the checkpoint of the model.
        checkpoint_path (Path): Path to the checkpoint of the model.
        hparams (dict): Hyperparameters of the models dataloader.
    """
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
