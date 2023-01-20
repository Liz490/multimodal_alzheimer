import os
import torch
from pkg.dataloader import MultiModalDataset
from torch.utils.data import DataLoader
from pkg.anat_cnn import Anat_CNN
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import math

from pkg.tabular_data.dl_model import Tabular_MRT_Model
from pkg.train_pet_cnn import ValidationLossTracker
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint

LOG_DIRECTORY = 'lightning_logs'
EXPERIMENT_NAME = 'optuna_tabular_mri_fusion_two_class'
EXPERIMENT_VERSION = None

# MRI model
BASEPATH = "/u/home/eisln/adlm_adni"
PATH_MRI_CNN = os.path.join(BASEPATH, 'lightning_logs/best_runs/mri_2_class/checkpoints/epoch=37-step=37.ckpt')

MODEL_MRI = Anat_CNN.load_from_checkpoint(PATH_MRI_CNN)


def options_list_to_dict(options: list) -> tuple[list, dict]:
    """
    Make options indexable by their string representation for optuna
    compatibility. Optuna needs hashable types for sampling.

    Args:
        options: List of options

    Returns:
        A list of the options converted to strings and a dict of
        the options with the strings as indices.
    """
    options_index = [str(o) for o in options]
    options_dict = {i: o for i, o in zip(options_index, options)}
    return options_index, options_dict


def optuna_objective(trial):
    """
    Defines options for hyperparameters and selects hyperparameters based on
    the options that performed well in previous runs. Starts training
    afterwards and returns the value of the objective function.

    See https://medium.com/optuna/using-optuna-to-optimize-pytorch-lightning-hyperparameters-d9e04a481585

    Args:
        trial: trial object, stores performance of hyperparameters
        from previous runs

    Returns:
        Validation loss of last epoch
    """

    # Define fixed parameters
    hparams = {
        'early_stopping_patience': 5,
        'max_epochs': 20,
        'path_mri': PATH_MRI_CNN,
        'n_classes': 2,
        'gpu_id': 2,
        'reduce_factor_lr_schedule': None
    }

    # Define hyperparameter options and ranges
    batch_size_options = [8, 16, 32, 64]
    l2_options = [0, 1e-1, 1e-2, 1e-3]
    lr_min = 1e-5
    lr_max = 1e-2
    gamma_options = [None, 1, 2, 5]

    # Let optuna select hyperparameters based on options defined above
    hparams['lr'] = trial.suggest_float('lr', lr_min, lr_max, log=True)
    hparams['batch_size'] = trial.suggest_categorical('batch_size',
                                                      batch_size_options)
    if hparams['batch_size'] >= 64:
        # Increase early stopping patience and maximum number of epochs,
        # if batch size is big
        hparams['early_stopping_patience'] = 10
        hparams['max_epochs'] = 50
    hparams['l2_reg'] = trial.suggest_categorical('l2_reg', l2_options)
    hparams['fl_gamma'] = trial.suggest_categorical('fl_gamma', gamma_options)

    # Train network
    try:
        val_loss = train_anat_tabular(hparams=hparams,
                                  model_mri=MODEL_MRI,
                                  experiment_name=EXPERIMENT_NAME,
                                  experiment_version=EXPERIMENT_VERSION)
        return val_loss
    except torch.cuda.OutOfMemoryError:
        print("Aborting run, not enough memory!")
        return math.inf


def train_anat_tabular(hparams, model_mri, experiment_name='', experiment_version=None):
    """
    Train model for MRI data.

    Args:
        hparams: A dictionary of hyperparameters
        experiment_name: Subdirectory for collection of runs in log directory
        experiment_version: Name of this specific run

    Returns:
        Validation loss of last epoch
    """
    pl.seed_everything(15, workers=True)

    # CALLBACKS
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    assert hparams['n_classes'] == 2 or hparams['n_classes'] == 3
    if hparams['n_classes'] == 2:
        binary_classification = True
    else:
        binary_classification = False

    # Setup datasets and dataloaders
    trainpath = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')
    valpath = os.path.join(os.getcwd(), 'data/val_path_data_labels.csv')

    trainset = MultiModalDataset(
        path=trainpath,
        modalities=['tabular', 't1w'],
        normalize_mri={'per_scan_norm': 'min_max'},
        binary_classification=binary_classification,
        quantile=model_mri.hparams['norm_percentile'])
    valset = MultiModalDataset(
        path=valpath,
        modalities=['tabular', 't1w'],
        normalize_mri={'per_scan_norm': 'min_max'},
        binary_classification=binary_classification,
        quantile=model_mri.hparams['norm_percentile'])

    trainloader = DataLoader(
        trainset,
        batch_size=hparams['batch_size'],
        shuffle=True,
        num_workers=32
    )

    valloader = DataLoader(
        valset,
        batch_size=hparams['batch_size'],
        shuffle=False,
        num_workers=32)

    # Get class distribution of the trainset for weighted loss
    _, weight_normalized = trainset.get_label_distribution()
    hparams['loss_class_weights'] = 1 - weight_normalized

    model = Tabular_MRT_Model(hparams=hparams)

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir='lightning_logs',
        name=experiment_name,
        version=experiment_version)

    val_loss_tracker = ValidationLossTracker()

    trainer = pl.Trainer(
        max_epochs=hparams['max_epochs'],
        logger=tb_logger,
        log_every_n_steps=5,
        accelerator='gpu',
        devices=1,
        callbacks=[
            EarlyStopping(
                monitor='val_loss_epoch',
                mode='min',
                patience=hparams['early_stopping_patience']
            ),
            val_loss_tracker,
            lr_monitor,
            ModelCheckpoint(monitor='val_loss_epoch')
        ]
    )

    trainer.fit(model, trainloader, valloader)
    return val_loss_tracker.val_loss[-1]


def optuna_optimization():
    """
    Create an optuna study and minimize the objective
    """
    study = optuna.create_study(direction="minimize")
    seconds_per_day = 24 * 60 * 60
    study.optimize(optuna_objective, n_trials=300, timeout=7 * seconds_per_day)


if __name__ == '__main__':
    #####################
    # Uncomment and comment the rest for optuna optimization
    # optuna_optimization()
    #####################

    # fine-tune best run (version 56)
    hparams = {
        'early_stopping_patience': 30,
        'max_epochs': 300,
        'norm_mean_train': 413.6510,
        'norm_std_train': 918.5371,
        'norm_mean_val': 418.4120,
        'norm_std_val': 830.2466,
        'n_classes': 2,
        'lr': 0.0008678312514285887,
        'batch_size': 32,
        'fl_gamma': 5,
        'ensemble_size': 4,
        'l2_reg': 0,
        'path_mri': '/u/home/eisln/adlm_adni/lightning_logs/best_runs/mri_2_class/checkpoints/epoch=37-step=37.ckpt',
        'reduce_factor_lr_schedule': 0.1
    }

    train_anat_tabular(hparams,
                   model_mri=MODEL_MRI,
                   experiment_name='best_runs',
                   experiment_version='2stage_tabular_mri_2_class')
