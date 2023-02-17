import os
import torch
import math
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from pkg.models.fusion_models.early_fusion import PET_MRI_EF, Random_Benchmark_All_CN
from pkg.models.fusion_models.anat_pet_featuremapfusion import PET_MRI_FMF
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pkg.utils.dataloader import MultiModalDataset
import sys

LOG_DIRECTORY = 'lightning_logs'
EXPERIMENT_NAME = 'optuna_maxout_3_class_feature_map_fusion'
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
    l2_options = [0, 1e-1, 1e-2, 1e-3]
    gamma_options = [None, 1, 2, 5]
    q_options = [0.95, 0.98, 0.99, 1]

    # Dynamic generation of conv_out options
    conv_out_first_options = [8, 16, 32]
    # IN COMPARISON TO THE SMALL CNN WE USE FEWER LAYERS, THEN FUSE THE FEATURE MAPS AND THEN ADD ANOTHER LAYER
    n_conv_options = [2, 3]
    conv_out_options = []
    for x in conv_out_first_options:
        for n in n_conv_options:
            layers = tuple([2**i * x for i in range(n)])
            conv_out_options.append(layers)
    
    # Make options indexable by their string representation for optuna compatibility
    conv_out_options_index = [str(o) for o in conv_out_options]
    conv_out_options_dict = {i: o for i, o in zip(conv_out_options_index,
                                                  conv_out_options)}

    filter_size_options = [(5, 5, 3),
                           (7, 5, 3),
                           (5, 5, 5),
                           (3, 3, 3)]
    
    n_layers_fusion_options = [1]
    filter_size_fusion_options = [3, 4, 5]

    # Make options indexable by their string representation for optuna compatibility
    filter_size_options_index = [str(o) for o in filter_size_options]
    filter_size_options_dict = {i: o for i, o in zip(filter_size_options_index,
                                                     filter_size_options)}

    linear_out_options = [64] #[False, 32, 64, 128]
    n_out_fusion_options = [64, 128, 256]
    dropout_conv_min = 0.05
    dropout_conv_max = 0.2
    dropout_dense_min = 0.2
    dropout_dense_max = 0.5
    batchnorm_options = [True, False]
    batchnorm_fusion_options = [True, False]

    # Let optuna configure hyperparameters based on options defined above
    hparams = {
        'early_stopping_patience': 5,
        'max_epochs': 20,
        'norm_mean': 0.5145,
        'norm_std': 0.5383,
        'reduce_factor_lr_schedule': None,
        'n_classes': 3,
        'best_k_checkpoints': 3,
        'fusion_mode': 'maxout'
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
    hparams['batchnorm_fusion'] = trial.suggest_categorical('batchnorm_fusion',
                                                     batchnorm_fusion_options)
    hparams['linear_out'] = trial.suggest_categorical(
        'linear_out', linear_out_options)    
    hparams['n_layers_fusion'] = trial.suggest_categorical('n_layers_fusion',
                                                      n_layers_fusion_options)
    hparams['filter_size_fusion'] = trial.suggest_categorical('filter_size_fusion',
                                                      filter_size_fusion_options)
    
    hparams['n_out_fusion'] = trial.suggest_categorical('n_out_fusion',
                                                      n_out_fusion_options)                                                  
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
    hparams['norm_percentile'] = trial.suggest_categorical('norm_percentile',
                                                           q_options)
    hparams['l2_reg'] = trial.suggest_categorical('l2_reg', l2_options)
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

    # CALLBACKS
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # TRANSFORMS
    normalization_pet = {'mean': hparams['norm_mean'], 'std': hparams['norm_std']}

    assert hparams['n_classes'] == 2 or hparams['n_classes'] == 3
    if hparams['n_classes'] == 2:
        binary_classification=True
        normalization_mri = {'per_scan_norm': 'min_max'} #{'all_scan_norm': {'mean': 426.9336, 'std': 1018.7830}}
    else:
        binary_classification=False
        normalization_mri = {'per_scan_norm': 'min_max'} #{'all_scan_norm': {'mean': 414.8254, 'std': 920.8566}}

    # DATASET AND DATALOADER
    trainpath = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')
    valpath = os.path.join(os.getcwd(), 'data/val_path_data_labels.csv')

    remove_mci = hparams["n_classes"] == 2
    
    trainset = MultiModalDataset(
        path=trainpath, 
        modalities=['pet1451', 't1w'], 
        normalize_mri={'per_scan_norm': 'min_max'},
        normalize_pet={'mean': hparams['norm_mean'], 'std': hparams['norm_std']},
        binary_classification=binary_classification, 
        quantile=hparams['norm_percentile'])
    valset = MultiModalDataset(
        path=valpath, 
        modalities=['pet1451', 't1w'], 
        normalize_mri=normalization_mri,
        normalize_pet = {'mean': hparams['norm_mean'], 'std': hparams['norm_std']},
        binary_classification=binary_classification, 
        quantile=hparams['norm_percentile'])



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
    hparams['loss_class_weights_human_readable'] = hparams['loss_class_weights'].tolist()  # original hparam is a Tensor that isn't stored in human readable format

    # model = Random_Benchmark(hparams=hparams)
    model = PET_MRI_FMF(hparams=hparams)
    print(model)
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
                patience=hparams['early_stopping_patience']),
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
    study = optuna.create_study(direction="minimize")
    study.optimize(optuna_just_sampling, n_trials=300, timeout=86400)


if __name__ == '__main__':
    #####################
    # Uncomment and comment the rest for optuna optimization
    # optuna_optimization()
    #####################

    # # Best two class concatenate v47:
    # hparams = {
    #     'early_stopping_patience': 30,
    #     'max_epochs': 300,
    #     'norm_mean': 0.5145,
    #     'norm_std': 0.5383,
    #     'lr': 3.0759705325385196e-05,
    #     'batch_size': 64,
    #     'conv_out': [16, 32],
    #     'filter_size': [7, 5, 3],
    #     'filter_size_fusion': 3,
    #     'n_classes': 2,
    #     'linear_out': 64,
    #     'fl_gamma': 2,
    #     'batchnorm': True,
    #     'batchnorm_fusion': True,
    #     'reduce_factor_lr_schedule': 0.1,
    #     'norm_percentile': 0.95,
    #     'best_k_checkpoints': 3,
    #     'fusion_mode': 'concatenate',
    #     'l2_reg': 0.01,
    #     'n_layers_fusion': 1,
    #     'n_out_fusion': 64

    # }

    # Best two class maxout v27:
    hparams = {
        'early_stopping_patience': 30,
        'max_epochs': 300,
        'norm_mean': 0.5145,
        'norm_std': 0.5383,
        'lr': 0.00020495402328039346,
        'batch_size': 64,
        'conv_out': [16, 32, 64],
        'filter_size': [5, 5, 5],
        'filter_size_fusion': 5,
        'n_classes': 2,
        'linear_out': 64,
        'fl_gamma': 5,
        'batchnorm': False,
        'batchnorm_fusion': False,
        'reduce_factor_lr_schedule': 0.1,
        'norm_percentile': 0.95,
        'best_k_checkpoints': 3,
        'fusion_mode': 'maxout',
        'l2_reg': 0,
        'n_layers_fusion': 1,
        'n_out_fusion': 64,
        'dropout_conv_p': 0.17474488076276287,
        'dropout_dense': 0.37401418387718594

    }

    
    train(hparams, experiment_name='testruns', experiment_version='maxout_2_class_feature_map_fusion_v27')

