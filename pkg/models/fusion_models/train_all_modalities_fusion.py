from pathlib import Path
import torch
from pkg.utils.dataloader import MultiModalDataset
from torch.utils.data import DataLoader
from pkg.models.mri_models.anat_cnn import Anat_CNN
from pkg.models.pet_models.pet_cnn import Small_PET_CNN
from pkg.models.fusion_models.all_modalities_fusion import All_Modalities_Fusion
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import math
from pkg.models.pet_models.train_pet_cnn import ValidationLossTracker
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pkg.models.fusion_models.anat_pet_fusion import Anat_PET_CNN
from pkg.models.fusion_models.tabular_mri_fusion import Tabular_MRT_Model
from pkg.models.fusion_models.pet_tabular_fusion import PET_TABULAR_CNN

# tensorboard and checkpoint logging
LOG_DIRECTORY = 'lightning_logs'
EXPERIMENT_NAME = 'testruns'
EXPERIMENT_VERSION = 'all_mods'

# PATH_PET_MRI = Path.cwd() / 'lightning_logs/best_runs/...'
# PATH_TAB_MRI = Path.cwd() / 'lightning_logs/best_runs/...'

# PET and MRI stage 1 models paths for normalization
# PET and MRI models
PATH_PET_CNN_2_CLASS = '/data2/practical-wise2223/adni/adni_1/lightning_checkpoints/lightning_logs/best_runs/pet_2_class/checkpoints/epoch=112-step=112.ckpt'
PATH_MRI_CNN_2_CLASS = '/data2/practical-wise2223/adni/adni_1/lightning_checkpoints/lightning_logs/best_runs/mri_2_class/checkpoints/epoch=37-step=37.ckpt'
PATH_PET_CNN_3_CLASS = '/u/home/eisln/adlm_adni/lightning_logs/pet_3_class_retrain_best/v204/checkpoints/epoch=28-step=28.ckpt'
PATH_MRI_CNN_3_CLASS = '/u/home/eisln/adlm_adni/lightning_logs/optuna_mri_3_class/version_48/checkpoints/epoch=32-step=32.ckpt'
# # load checkpoints
# MODEL_PET_2_CLASS = Small_PET_CNN.load_from_checkpoint(PATH_PET_CNN_2_CLASS)
# MODEL_MRI_2_CLASS = Anat_CNN.load_from_checkpoint(PATH_MRI_CNN_2_CLASS)
# MODEL_PET_3_CLASS = Small_PET_CNN.load_from_checkpoint(PATH_PET_CNN_3_CLASS)
# MODEL_MRI_3_CLASS = Anat_CNN.load_from_checkpoint(PATH_MRI_CNN_3_CLASS)
# PATH_PET_CNN = Path.cwd() / 'lightning_logs/best_runs/pet_2_class/checkpoints/epoch=112-step=112.ckpt'
# PATH_MRI_CNN = Path.cwd() / 'lightning_logs/best_runs/mri_2_class/checkpoints/epoch=37-step=37.ckpt'
# load checkpoints
# MODEL_PET = Small_PET_CNN.load_from_checkpoint(PATH_PET_CNN)
# MODEL_MRI = Anat_CNN.load_from_checkpoint(PATH_MRI_CNN)

# PET and MRI models

PATH_PET_MRI_2_CLASS = '/data2/practical-wise2223/adni/adni_1/lightning_checkpoints/lightning_logs/best_runs/2stage_pet_mri_2_class/checkpoints/epoch=26-step=26.ckpt'
PATH_PET_TAB_2_CLASS = '/vol/chameleon/users/schmiere/Documents/Code /adlm_adni/lightning_logs/optuna_pet_tabular_fusion_two_class/version_11/checkpoints/epoch=42-step=42.ckpt'
PATH_MRI_TAB_2_CLASS = '/vol/chameleon/users/schmiere/Documents/Code /adlm_adni/lightning_logs/tabular_mri_fusion_two_class/version_11/checkpoints/epoch=49-step=49.ckpt'
PATH_PET_MRI_3_CLASS = '/u/home/eisln/adlm_adni/lightning_logs/best_pet_mri_3_class_unfrozen/v28/checkpoints/epoch=32-val_f1=0.685.ckpt'
PATH_PET_TAB_3_CLASS = '/vol/chameleon/users/schmiere/Documents/Code /adlm_adni/lightning_logs/optuna_unfrozen_pet_tabular_fusion_two_class/version_30/checkpoints/epoch=11-val_f1=0.809.ckpt'
PATH_MRI_TAB_3_CLASS = '/u/home/eisln/adlm_adni/lightning_logs/best_runs/2stage_tabular_mri_3_class/checkpoints/epoch=21-val_f1=0.528.ckpt'

# MODEL_PET_MRI_2_CLASS = Anat_PET_CNN.load_from_checkpoint(PATH_PET_MRI_2_CLASS)
# MODEL_PET_TAB_2_CLASS = ''
# MODEL_MRI_TAB_2_CLASS = ''
# MODEL_PET_MRI_3_CLASS = Anat_PET_CNN.load_from_checkpoint(PATH_PET_MRI_3_CLASS)
# MODEL_PET_TAB_3_CLASS = ''
# MODEL_MRI_TAB_3_CLASS = Tabular_MRT_Model.load_from_checkpoint(PATH_MRI_TAB_3_CLASS)
# load checkpoints

# MODEL_PET_2_CLASS = Small_PET_CNN.load_from_checkpoint(PATH_PET_CNN_2_CLASS)
# MODEL_MRI_2_CLASS = Anat_CNN.load_from_checkpoint(PATH_MRI_CNN_2_CLASS)
# MODEL_PET_3_CLASS = Small_PET_CNN.load_from_checkpoint(PATH_PET_CNN_3_CLASS)
# MODEL_MRI_3_CLASS = Anat_CNN.load_from_checkpoint(PATH_MRI_CNN_3_CLASS)


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
        # 'path_anat_pet': PATH_PET_MRI,
        # 'path_anat_tab': PATH_TAB_MRI,
        # 'path_pet_tab': PATH_PET_TAB,
        'n_classes': 2,
        'gpu_id': 2,
        'reduce_factor_lr_schedule': None,
        'best_k_checkpoints': 3
    }

    # Define hyperparameter options and ranges
    batch_size_options = [8, 16, 32, 64]
    l2_options = [0, 1e-1, 1e-2, 1e-3]
    lr_min = 1e-5
    lr_max = 1e-2
    lr_pretrained_min = 1e-7
    lr_pretrained_max = 1e-5
    gamma_options = [None, 1, 2, 5]

    # Let optuna select hyperparameters based on options defined above
    hparams['lr'] = trial.suggest_float('lr', lr_min, lr_max, log=True)
    if hparams['n_classes'] == 3:
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
        val_loss = train(hparams=hparams,
                         experiment_name=EXPERIMENT_NAME,
                         experiment_version=EXPERIMENT_VERSION)
        return val_loss
    except torch.cuda.OutOfMemoryError:
        print("Aborting run, not enough memory!")
        return math.inf


def train(hparams, experiment_name='', experiment_version=None):
    # fix random seeds for reproducability
    pl.seed_everything(15, workers=True)

    # CALLBACKS
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    assert hparams['n_classes'] == 2 or hparams['n_classes'] == 3
    if hparams['n_classes'] == 2:
        binary_classification = True
        # 2 stage paths
        hparams['path_anat_pet'] = PATH_PET_MRI_2_CLASS
        hparams['path_anat_tab'] = PATH_MRI_TAB_2_CLASS
        hparams['path_pet_tab'] = PATH_PET_TAB_2_CLASS
        # 1 stage paths
        hparams['path_pet'] = PATH_PET_CNN_2_CLASS
        hparams['path_anat'] = PATH_MRI_CNN_2_CLASS
        # load 1 stage models for normalization
        MODEL_PET = Small_PET_CNN.load_from_checkpoint(PATH_PET_CNN_2_CLASS)
        MODEL_MRI = Anat_CNN.load_from_checkpoint(PATH_MRI_CNN_2_CLASS)
    else:
        binary_classification = False
        # 2 stage paths
        hparams['path_anat_pet'] = PATH_PET_MRI_3_CLASS
        hparams['path_anat_tab'] = PATH_MRI_TAB_3_CLASS
        hparams['path_pet_tab'] = PATH_PET_TAB_3_CLASS
        # 1 stage paths
        hparams['path_pet'] = PATH_PET_CNN_3_CLASS
        hparams['path_anat'] = PATH_MRI_CNN_3_CLASS
        # load 1 stage models for normalization
        MODEL_PET = Small_PET_CNN.load_from_checkpoint(PATH_PET_CNN_3_CLASS)
        MODEL_MRI = Anat_CNN.load_from_checkpoint(PATH_MRI_CNN_3_CLASS)

    # Setup datasets and dataloaders
    trainpath = Path.cwd() / 'data/train_path_data_labels.csv'
    valpath = Path.cwd() / 'data/val_path_data_labels.csv'

    trainset = MultiModalDataset(
        path=trainpath,
        modalities=['pet1451', 't1w', 'tabular'],
        normalize_mri={'per_scan_norm': 'min_max'},
        normalize_pet={'mean': MODEL_PET.hparams['norm_mean'],
                       'std': MODEL_PET.hparams['norm_std']},
        binary_classification=binary_classification,
        quantile=MODEL_MRI.hparams['norm_percentile'])
    valset = MultiModalDataset(
        path=valpath,
        modalities=['pet1451', 't1w', 'tabular'],
        normalize_mri={'per_scan_norm': 'min_max'},
        normalize_pet={'mean': MODEL_PET.hparams['norm_mean'],
                       'std': MODEL_PET.hparams['norm_std']},
        binary_classification=binary_classification,
        quantile=MODEL_MRI.hparams['norm_percentile'])

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

    model = All_Modalities_Fusion(hparams=hparams)

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
    study.optimize(optuna_objective, n_trials=300, timeout=7*seconds_per_day)


if __name__ == '__main__':
    #####################
    # Uncomment and comment the rest for optuna optimization
    optuna_optimization()
    #####################

    # # fine-tune best run (version 56)
    # hparams = {
    #     'early_stopping_patience': 30,
    #     'max_epochs': 300,
    #     'norm_mean_train': 413.6510,
    #     'norm_std_train': 918.5371,
    #     'norm_mean_val': 418.4120,
    #     'norm_std_val': 830.2466,
    #     'n_classes': 2,
    #     'lr': 0.0008678312514285887,
    #     'batch_size': 32,
    #     'fl_gamma': 5,
    #     'l2_reg': 0,
    #     'path_anat_pet': '...',
    #     'path_anat_tab': '...',
    #     'reduce_factor_lr_schedule': 0.1
    # }

    # train(hparams, experiment_name='best_runs')
