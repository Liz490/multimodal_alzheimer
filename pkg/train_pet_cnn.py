import os
import torch
import math
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from pet_cnn import Small_PET_CNN, Random_Benchmark_All_CN
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback
from dataloader import MultiModalDataset

LOG_DIRECTORY = 'lightning_logs'
EXPERIMENT_NAME = 'optuna_two_class'
EXPERIMENT_VERSION = None


class ValidationLossTracker(Callback):
    """
    Tracks validation loss per epoch across epochs

    See
    https://stackoverflow.com/questions/69276961/how-to-extract-loss-and-accuracy-from-logger-by-each-epoch-in-pytorch-lightning"""

    def __init__(self):
        self.val_loss = []

    def on_validation_epoch_end(self, trainer, module):
        val_loss = trainer.logged_metrics['val_loss_epoch']
        self.val_loss.append(val_loss)


def optuna_just_sampling(trial):
    """
    See https://medium.com/optuna/using-optuna-to-optimize-pytorch-lightning-hyperparameters-d9e04a481585
    """
    # Define hyperparameter options
    lr_min = 5e-6
    lr_max = 1e-3
    batch_size_options = [8, 16, 32, 64]
    gamma_options = [None, 1, 2, 5]

    # Dynamic generation of conv_out options
    conv_out_first_options = [8, 16, 32]
    n_conv_options = [3, 4]
    conv_out_options = []
    for x in conv_out_first_options:
        for n in n_conv_options:
            layers = tuple([2**i * x for i in range(n)])
            conv_out_options.append(layers)

    # Make options indexable by their string representation for optuna compatibility
    conv_out_options_index = [str(o) for o in conv_out_options]
    conv_out_options_dict = {i: o for i, o in zip(conv_out_options_index,
                                                  conv_out_options)}

    filter_size_options = [(5, 5, 3, 3),
                           (7, 5, 3, 3),
                           (5, 5, 5, 3),
                           (3, 3, 3, 3)]

    # Make options indexable by their string representation for optuna compatibility
    filter_size_options_index = [str(o) for o in filter_size_options]
    filter_size_options_dict = {i: o for i, o in zip(filter_size_options_index,
                                                     filter_size_options)}

    linear_out_options = [False, 32, 64, 128]
    dropout_conv_min = 0.05
    dropout_conv_max = 0.2
    dropout_dense_min = 0.2
    dropout_dense_max = 0.5
    batchnorm_options = [True, False]

    # Let optuna configure hyperparameters based on options defined above
    hparams = {
        'early_stopping_patience': 5,
        'max_epochs': 20,
        'norm_mean': 0.5145,
        'norm_std': 0.5383,
        'n_classes': 2
    }
    hparams['lr'] = trial.suggest_float(
        'learning_rate', lr_min, lr_max, log=True)
    conv_out_idx = trial.suggest_categorical('conv_out',
                                             conv_out_options_index)
    hparams['conv_out'] = conv_out_options_dict[conv_out_idx]
    filter_size_idx = trial.suggest_categorical(
        'filter_size', filter_size_options_index)
    hparams['filter_size'] = filter_size_options_dict[filter_size_idx]
    hparams['batchnorm'] = trial.suggest_categorical('batchnorm',
                                                     batchnorm_options)
    hparams['linear_out'] = trial.suggest_categorical(
        'linear_out', linear_out_options)
    hparams['batch_size'] = trial.suggest_categorical('batch_size',
                                                      batch_size_options)
    if hparams['batch_size'] >= 64:
        hparams['early_stopping_patience'] = 10
        hparams['max_epochs'] = 50
    dropout_conv = trial.suggest_categorical('dropout_conv', (True, False))
    dropout_dense = trial.suggest_categorical('dropout_dense', (True, False))
    if dropout_conv:
        hparams['dropout_conv_p'] = trial.suggest_float(
            'dropout_conv_p', dropout_conv_min, dropout_conv_max)
    if dropout_dense:
        hparams['dropout_dense'] = trial.suggest_float(
            'dropout_dense_p', dropout_dense_min, dropout_dense_max)

    hparams['fl_gamma'] = trial.suggest_categorical('fl_gamma', gamma_options)
    # Train network
    try:
        val_loss = train(hparams,
                         experiment_name=EXPERIMENT_NAME,
                         experiment_version=EXPERIMENT_VERSION)
        return val_loss
    except torch.cuda.OutOfMemoryError:
        print("Aborting run, not enough memory!")
        return math.inf


def train(hparams,
          experiment_name='',
          experiment_version=None,
          trial=None):
    pl.seed_everything(5, workers=True)

    # TRANSFORMS
    normalization_pet = {'mean': hparams['norm_mean'], 'std': hparams['norm_std']}

    assert hparams['n_classes'] == 2 or hparams['n_classes'] == 3
    if hparams['n_classes'] == 2:
        binary_classification=True
    else:
        binary_classification=False

    # DATASET AND DATALOADER
    trainpath = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')
    valpath = os.path.join(os.getcwd(), 'data/val_path_data_labels.csv')

    remove_mci = hparams["n_classes"] == 2
    trainset = MultiModalDataset(path=trainpath, 
                                modalities=['pet1451'],
                                normalize_pet=normalization_pet,
                                binary_classification=binary_classification)
    valset = MultiModalDataset(path=valpath,
                            modalities=['pet1451'],
                            normalize_pet=normalization_pet,
                            binary_classification=binary_classification)



    trainloader = DataLoader(
        trainset,
        batch_size=hparams['batch_size'],
        shuffle=True,
        num_workers=32
    )
    valloader = DataLoader(
        valset,
        batch_size=hparams['batch_size'],
        num_workers=32)

    _, weight_normalized = trainset.get_label_distribution()
    hparams['loss_class_weights'] = 1 - weight_normalized

    # model = Random_Benchmark(hparams=hparams)
    model = Small_PET_CNN(hparams=hparams)

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
                monitor='val_loss',
                mode='min',
                patience=hparams['early_stopping_patience']),
            val_loss_tracker
        ]
    )

    trainer.fit(model, trainloader, valloader)
    return val_loss_tracker.val_loss[-1]


def optuna_optimization():
    study = optuna.create_study(direction="minimize")
    study.optimize(optuna_just_sampling, n_trials=300, timeout=86400)


if __name__ == '__main__':
    #####################
    # Uncomment and comment the rest for optuna optimization
    # optuna_optimization()
    #####################

    # Experimental
    # hparams = {
    #     'early_stopping_patience': 5,
    #     'max_epochs': 20,
    #     'norm_mean': 0.5145,
    #     'norm_std': 0.5383
    # }

    # hparams['lr'] = 0.00006
    # hparams['batch_size'] = 8
    # hparams['conv_out'] = [8, 16, 32, 64]
    # hparams['filter_size'] = [5, 5, 5, 3]  # More filters for more layers!
    # hparams['batchnorm'] = True
    # # hparams['dropout_conv_p'] = 0.1
    # # hparams['dropout_dense_p'] = 0.5
    # hparams['linear_out'] = 64
    # hparams["n_classes"] = 2


    # Best two class:
    hparams = {
        'early_stopping_patience': 10,
        'max_epochs': 50,
        'norm_mean': 0.5145,
        'norm_std': 0.5383,
        'lr': 0.00075356,
        'batch_size': 64,
        'conv_out': [8, 16, 32, 64],
        'filter_size': [3, 3, 3, 3],
        'batchnorm': False,
        'n_classes': 2,
        'linear_out': 64
    }

    # Best two class with batchnorm
    # hparams = {
    #     'early_stopping_patience': 10,
    #     'max_epochs': 50,
    #     'norm_mean': 0.5145,
    #     'norm_std': 0.5383,
    #     'lr': 0.00075356,
    #     'batch_size': 64,
    #     'conv_out': [8, 16, 32, 64],
    #     'filter_size': [3, 3, 3, 3],
    #     'batchnorm': True,
    #     'n_classes': 2,
    #     'linear_out': 64
    # }

    # train(hparams, experiment_name='', experiment_version='unchanged')
    train(hparams)
    # TODO rerun with MCI samples
