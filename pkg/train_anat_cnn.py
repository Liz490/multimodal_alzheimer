import os
from numpy import float64
import torch
from torchaudio import transforms
from dataloader import MultiModalDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from sklearn.metrics import f1_score
from pkg.anat_cnn import Anat_CNN
import optuna
# from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import sys
import math
from pkg.train_pet_cnn import MetricTracker

LOG_DIRECTORY = 'lightning_logs'
EXPERIMENT_NAME = 'optuna_mri_two_class_var_resnet'
EXPERIMENT_VERSION = None


def optuna_just_sampling(trial):
    """
    See https://medium.com/optuna/using-optuna-to-optimize-pytorch-lightning-hyperparameters-d9e04a481585
    """
    hparams = {
        'early_stopping_patience': 5,
        'max_epochs': 20,
        'norm_mean_train': 413.6510,
        'norm_std_train': 918.5371,
        'norm_mean_val': 418.4120,
        'norm_std_val': 830.2466,
        'n_classes': 2,
        'gpu_id': 6,
    }

    # Define hyperparameter options
    batch_size_options = [8, 16, 32, 64]

    # # Dynamic generation of conv_out options
    # conv_out_first_options = [256, 128, 64, 32, 16, 8]
    # n_conv_options = [0, 2]
    # conv_out_options = []
    # for x in conv_out_first_options:
    #     for n in n_conv_options:
    #         layers = tuple([int(x / 2**i) for i in range(n)])
    #         conv_out_options.append(layers)

    # # Make options indexable by their string representation for optuna
    # # compatibility
    # conv_out_options_index = [str(o) for o in conv_out_options]
    # conv_out_options_dict = {i: o for i, o in zip(conv_out_options_index,
    #                                               conv_out_options)}

    # filter_size_options = [(5, 5), (5, 3), (3, 3)]

    # # Make options indexable by their string representation for optuna
    # # compatibility
    # filter_size_options_index = [str(o) for o in filter_size_options]
    # filter_size_options_dict = {i: o for i, o in zip(filter_size_options_index,
    #                                                  filter_size_options)}

    # Dynamic generation of dense_out options
    dense_out_first_options = [256, 128, 64]
    n_dense_options = [0, 3]
    dense_out_options = []
    for x in dense_out_first_options:
        for n in n_dense_options:
            layers = tuple([x for _ in range(n)])
            dense_out_options.append(layers)
            layers_shrinking = tuple([int(x / 2**i) for i in range(n)])
            dense_out_options.append(layers_shrinking)

    # Make options indexable by their string representation for optuna
    # compatibility
    dense_out_options_index = [str(o) for o in dense_out_options]
    dense_out_options_dict = {i: o for i, o in zip(dense_out_options_index,
                                                   dense_out_options)}

    l2_options = [0, 1e-1, 1e-2, 1e-3]
    lr_min = 1e-5
    lr_max = 1e-2
    lr_pretrained_min = 1e-7
    lr_pretrained_max = 1e-5
    q_options = [0.95, 0.98, 0.99, 1]
    gamma_options = [None, 1, 2, 5]
    resnet_options = [10, 18, 50]

    # Let optuna configure hyperparameters based on options defined above
    hparams['lr'] = trial.suggest_float(
        'lr', lr_min, lr_max, log=True)
    freeze = trial.suggest_categorical('freeze', (True, False))
    if not freeze:
        hparams['lr_pretrained'] = trial.suggest_float(
            'lr_pretrained', lr_pretrained_min, lr_pretrained_max, log=True)
    else:
        hparams['lr_pretrained'] = None
    # conv_out_idx = trial.suggest_categorical('conv_out',
    #                                          conv_out_options_index)
    # hparams['conv_out'] = conv_out_options_dict[conv_out_idx]
    hparams['conv_out'] = []
    # filter_size_idx = trial.suggest_categorical(
    #     'filter_size', filter_size_options_index)
    # hparams['filter_size'] = filter_size_options_dict[filter_size_idx]
    hparams['filter_size'] = []
    # hparams['batchnorm_conv'] = trial.suggest_categorical('batchnorm_conv',
    #                                                       (True, False))
    hparams['batchnorm_begin'] = trial.suggest_categorical('batchnorm_begin',
                                                           (True, False))
    hparams['batchnorm_dense'] = trial.suggest_categorical('batchnorm_dense',
                                                           (True, False))
    hparams['batch_size'] = trial.suggest_categorical('batch_size',
                                                      batch_size_options)
    if hparams['batch_size'] >= 64:
        hparams['early_stopping_patience'] = 10
        hparams['max_epochs'] = 50
    hparams['l2_reg'] = trial.suggest_categorical('l2_reg', l2_options)
    dense_out_idx = trial.suggest_categorical('linear_out',
                                              dense_out_options_index)
    hparams['linear_out'] = dense_out_options_dict[dense_out_idx]
    hparams['norm_percentile'] = trial.suggest_categorical('norm_percentile',
                                                           q_options)
    hparams['fl_gamma'] = trial.suggest_categorical('fl_gamma', gamma_options)
    hparams['resnet_depth'] = trial.suggest_categorical('resnet_depth', resnet_options)

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
          experiment_version=None):
    pl.seed_everything(15, workers=True)

    # DATASET AND DATALOADER
    trainpath = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')
    valpath = os.path.join(os.getcwd(), 'data/val_path_data_labels.csv')

    trainset = MultiModalDataset(
        path=trainpath, modalities=['t1w'], per_scan_norm='min_max',
        binary_classification=True, q=hparams['norm_percentile'])
    valset = MultiModalDataset(
        path=valpath, modalities=['t1w'], per_scan_norm='min_max',
        binary_classification=True, q=hparams['norm_percentile'])

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

    _, weight_normalized = trainset.get_label_distribution()
    hparams['loss_class_weights'] = 1 - weight_normalized
    print(1 - weight_normalized)

    # sys.exit()
    model = Anat_CNN(hparams=hparams)

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir='lightning_logs',
        name=experiment_name,
        version=experiment_version)

    mt_cb = MetricTracker()
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
                patience=hparams['early_stopping_patience']
            ),
            mt_cb
        ]
    )

    trainer.fit(model, trainloader, valloader)
    return mt_cb.val_loss[-1]


def optuna_optimization():
    study = optuna.create_study(direction="minimize")
    day = 86400
    study.optimize(optuna_just_sampling, n_trials=400, timeout=7*day)


if __name__ == '__main__':
    optuna_optimization()

    # hparams = {
    #     'early_stopping_patience': 5,
    #     'max_epochs': 20,
    #     'norm_mean_train': 413.6510,
    #     'norm_std_train': 918.5371,
    #     'norm_mean_val': 418.4120,
    #     'norm_std_val': 830.2466,
    #     'n_classes': 2,
    #     'lr': 1e-4,
    #     'batch_size': 64,
    #     'fl_gamma': 2,
    #     # 'conv_out': [],
    #     # 'filter_size': [5, 5],
    #     'lr_pretrained': 1e-5,
    #     'batchnorm_begin': True,
    #     # 'batchnorm_conv': True,
    #     'batchnorm_dense': True,
    #     'l2_reg': 1e-2,
    #     # 'linear_out': [256, 256, 256],
    #     'linear_out': [],
    #     'norm_percentile': 0.99,
    #     'resnet_depth': 18,
    #     'gpu_id': 6,
    # }

    # train(hparams)
