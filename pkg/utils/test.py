from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pkg.utils.dataloader import MultiModalDataset


def test(testset: MultiModalDataset,
         model: pl.LightningModule,
         experiment_name: str) -> None:
    """
    Test a model on the test set. Logs results to tensorboard under
    the name of the checkpoints parent directory. I.e.
    lightning_logs/test_set_<name of checkpoint parent directory>.

    Args:
        testset (MultiModalDataset): Test set.
        model (pl.LightningModule): Hyperparameters of the models dataloader.
        experiment_name (str): Name of the log directory.
    """
    pl.seed_everything(5, workers=True)
    hparams = model.hparams

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
