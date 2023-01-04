import os
import torch
from dataloader import MultiModalDataset
from torch.utils.data import DataLoader
from pkg.pet_resnet_cnn import PET_CNN_ResNet
import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import math
from pkg.train_pet_cnn import ValidationLossTracker

LOG_DIRECTORY = 'lightning_logs'
EXPERIMENT_NAME = 'optuna_pet_two_class_var_resnet'
EXPERIMENT_VERSION = None


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
        'norm_mean': 0.5145,
        'norm_std': 0.5383,
        'n_classes': 2,
        'gpu_id': 3,
    }

    def generate_linear_block_options(first_layer_options: list,
                                      n_layers_options: list) -> list[tuple]:
        """
        Generate options for the shape of the linear classification layers.

        Args:
            first_layer_options: Options for the size of the first layer
            n_layers_options: Options for the number of layers

        Returns:
            List of tuples representing the layer sizes of the dense
            block at the end of the network
        """
        dense_out_options = []
        for x in first_layer_options:
            for n in n_layers_options:
                # Add option with the same size for all layers
                layers = tuple([x for _ in range(n)])
                dense_out_options.append(layers)

                # Add option with each layer half the size of the previous
                layers_shrinking = tuple([int(x / 2**i) for i in range(n)])
                dense_out_options.append(layers_shrinking)
        return dense_out_options

    # Define hyperparameter options and ranges
    batch_size_options = [8, 16, 32, 64]
    l2_options = [0, 1e-1, 1e-2, 1e-3]
    lr_min = 1e-5
    lr_max = 1e-2
    lr_pretrained_min = 1e-7
    lr_pretrained_max = 1e-5
    gamma_options = [None, 1, 2, 5]
    resnet_options = [10, 18, 50]
    dense_out_options_index, dense_out_options_dict = options_list_to_dict(
        generate_linear_block_options([256, 128, 64], [0, 3])
    )

    # Let optuna select hyperparameters based on options defined above
    hparams['lr'] = trial.suggest_float('lr', lr_min, lr_max, log=True)
    freeze = trial.suggest_categorical('freeze', (True, False))
    if not freeze:
        # Only set lr_pretrained if optuna selected freeze=False
        hparams['lr_pretrained'] = trial.suggest_float(
            'lr_pretrained', lr_pretrained_min, lr_pretrained_max, log=True)
    else:
        hparams['lr_pretrained'] = None
    hparams['conv_out'] = []
    hparams['filter_size'] = []
    hparams['batchnorm_begin'] = trial.suggest_categorical('batchnorm_begin',
                                                           (True, False))
    hparams['batchnorm_dense'] = trial.suggest_categorical('batchnorm_dense',
                                                           (True, False))
    hparams['batch_size'] = trial.suggest_categorical('batch_size',
                                                      batch_size_options)
    if hparams['batch_size'] >= 64:
        # Increase early stopping patience and maximum number of epochs,
        # if batch size is big
        hparams['early_stopping_patience'] = 10
        hparams['max_epochs'] = 50
    hparams['l2_reg'] = trial.suggest_categorical('l2_reg', l2_options)
    hparams['fl_gamma'] = trial.suggest_categorical('fl_gamma', gamma_options)
    hparams['resnet_depth'] = trial.suggest_categorical('resnet_depth',
                                                        resnet_options)

    # For shape of linear layers let optuna select the string representation
    # of the option and then set the hyperparameter dictionary entry based on
    # this.
    dense_out_idx = trial.suggest_categorical('linear_out',
                                              dense_out_options_index)
    hparams['linear_out'] = dense_out_options_dict[dense_out_idx]

    # Train network
    try:
        val_loss = train_pet_resnet(hparams,
                              experiment_name=EXPERIMENT_NAME,
                              experiment_version=EXPERIMENT_VERSION)
        return val_loss
    except torch.cuda.OutOfMemoryError:
        print("Aborting run, not enough memory!")
        return math.inf


def train_pet_resnet(hparams, experiment_name='', experiment_version=None):
    """
    Train model for PET data.

    Args:
        hparams: A dictionary of hyperparameters
        experiment_name: Subdirectory for collection of runs in log directory
        experiment_version: Name of this specific run

    Returns:
        Validation loss of last epoch
    """
    pl.seed_everything(15, workers=True)

    # TRANSFORMS
    normalization_pet = {'mean': hparams['norm_mean'], 'std': hparams['norm_std']}

    # Setup datasets and dataloaders
    trainpath = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')
    valpath = os.path.join(os.getcwd(), 'data/val_path_data_labels.csv')

    trainset = MultiModalDataset(path=trainpath, 
                                modalities=['pet1451'],
                                normalize_pet=normalization_pet,
                                binary_classification=True)
    valset = MultiModalDataset(path=valpath,
                            modalities=['pet1451'],
                            normalize_pet=normalization_pet,
                            binary_classification=True)

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

    model = PET_CNN_ResNet(hparams=hparams)

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
                patience=hparams['early_stopping_patience']
            ),
            val_loss_tracker
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

    # # Experimental hyperparameters
    # hparams = {
    #     'early_stopping_patience': 5,
    #     'max_epochs': 20,
    #     'norm_mean': 0.5145,
    #     'norm_std': 0.5383,
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
    #     'resnet_depth': 18,
    #     'gpu_id': 3,
    # }

    # train_pet_resnet(hparams)
