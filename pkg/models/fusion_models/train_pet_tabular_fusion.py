import os
import torch
from pkg.utils.dataloader import MultiModalDataset
from torch.utils.data import DataLoader
from pkg.models.pet_models.pet_cnn import Small_PET_CNN
from pet_tabular_fusion import PET_TABULAR_CNN
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import math
from pkg.models.pet_models.train_pet_cnn import ValidationLossTracker
import sys
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint

# tensorboard and checkpoint logging
LOG_DIRECTORY = 'lightning_logs'
EXPERIMENT_NAME = 'optuna_unfrozen_pet_tabular_fusion_three_class'
EXPERIMENT_VERSION = None

# PET and MRI models
BASEPATH = '/u/home/eisln/adlm_adni'
PATH_PET_CNN_2_CLASS = '/data2/practical-wise2223/adni/adni_1/lightning_checkpoints/lightning_logs/best_runs/pet_2_class/checkpoints/epoch=112-step=112.ckpt'
PATH_PET_CNN_3_CLASS = os.path.join(BASEPATH, 'lightning_logs/pet_3_class_retrain_best/v204/checkpoints/epoch=28-step=28.ckpt')

# load checkpoints
MODEL_PET_2_CLASS = Small_PET_CNN.load_from_checkpoint(PATH_PET_CNN_2_CLASS)
MODEL_PET_3_CLASS = Small_PET_CNN.load_from_checkpoint(PATH_PET_CNN_3_CLASS)


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
        'n_classes': 3,
        'gpu_id': 2,
        'reduce_factor_lr_schedule': None,
        'ensemble_size': 4,
        'best_k_checkpoints': 3
    }


    hparams['path_pet'] = PATH_PET_CNN_2_CLASS if hparams['n_classes'] == 2 else PATH_PET_CNN_3_CLASS


    # Define hyperparameter options and ranges
    batch_size_options = [8, 16, 32, 64]
    l2_options = [0, 1e-1, 1e-2, 1e-3]
    lr_min = 1e-5
    lr_max = 1e-2
    lr_pretrained_min = 1e-7
    lr_pretrained_max = 1e-5
    gamma_options = [None, 1, 2, 5]
    dim_red_options= [True, False]


    # Let optuna select hyperparameters based on options defined above
    hparams['simple_dim_red'] = trial.suggest_categorical('simple_dim_red', dim_red_options)
    hparams['lr'] = trial.suggest_float('lr', lr_min, lr_max, log=True)
    if hparams['n_classes']==3:
        # Only set lr_pretrained if optuna selected freeze=False
        hparams['lr_pretrained'] = trial.suggest_float(
            'lr_pretrained', lr_pretrained_min, lr_pretrained_max, log=True)
    else:
        hparams['lr_pretrained'] = None
    hparams['batch_size'] = trial.suggest_categorical('batch_size',
                                                      batch_size_options)
    if hparams['batch_size'] >= 64:
        # Increase early stopping patience and maximum number of epochs,
        # if batch size is big
        hparams['early_stopping_patience'] = 10
        hparams['max_epochs'] = 50
    hparams['l2_reg'] = trial.suggest_categorical('l2_reg', l2_options)
    # gamma parameter for focal loss
    hparams['fl_gamma'] = trial.suggest_categorical('fl_gamma', gamma_options)

    # Train network
    try:
        val_loss = train_pet_tabular(hparams=hparams,
                                  experiment_name=EXPERIMENT_NAME,
                                  experiment_version=EXPERIMENT_VERSION)
        return val_loss
    except torch.cuda.OutOfMemoryError:
        print("Aborting run, not enough memory!")
        return math.inf


def train_pet_tabular(hparams, experiment_name='', experiment_version=None):
    """
    Train model for MRI data.

    Args:
        hparams: A dictionary of hyperparameters
        experiment_name: Subdirectory for collection of runs in log directory
        experiment_version: Name of this specific run

    Returns:
        Validation loss of last epoch
    """
    # fix random seeds for reproducability
    pl.seed_everything(15, workers=True)

    # CALLBACKS
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    assert hparams['n_classes'] == 2 or hparams['n_classes'] == 3
    if hparams['n_classes'] == 2:
        binary_classification = True
        model_pet = MODEL_PET_2_CLASS
    else:
        binary_classification = False
        model_pet = MODEL_PET_3_CLASS

    # Setup datasets and dataloaders
    trainpath = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')
    valpath = os.path.join(os.getcwd(), 'data/val_path_data_labels.csv')

    trainset = MultiModalDataset(
        path=trainpath,
        modalities=['pet1451', 'tabular'],
        normalize_pet={'mean': model_pet.hparams['norm_mean'], 'std': model_pet.hparams['norm_std']},
        binary_classification=binary_classification)
    valset = MultiModalDataset(
        path=valpath,
        modalities=['pet1451', 'tabular'],
        normalize_pet={'mean': model_pet.hparams['norm_mean'], 'std': model_pet.hparams['norm_std']},
        binary_classification=binary_classification)

    trainloader = DataLoader(
        trainset,
        batch_size=hparams['batch_size'],
        shuffle=True,
        num_workers=32,
        drop_last=True
    )

    valloader = DataLoader(
        valset,
        batch_size=hparams['batch_size'],
        shuffle=False,
        num_workers=32,
        drop_last=True
    )

    # Get class distribution of the trainset for weighted loss
    _, weight_normalized = trainset.get_label_distribution()
    hparams['loss_class_weights'] = 1 - weight_normalized

    model = PET_TABULAR_CNN(hparams=hparams)

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
            ModelCheckpoint(monitor='val_loss_epoch',
                            save_top_k=hparams['best_k_checkpoints'],
                            mode='min',
                            filename='epoch={epoch}-val_loss={val_loss_epoch:.3f}',
                            auto_insert_metric_name=False),
            ModelCheckpoint(monitor='val_f1_epoch',
                        save_top_k=hparams['best_k_checkpoints'],
                        mode='max',
                        filename='epoch={epoch}-val_f1={val_f1_epoch:.3f}',
                        auto_insert_metric_name=False)
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
    optuna_optimization()
