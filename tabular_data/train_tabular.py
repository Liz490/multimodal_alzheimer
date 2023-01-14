import os
import torch
from pkg.dataloader import MultiModalDataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import math
from dl_model import tabularModel

def train_tabular(hparams, binary_classification = True):
    # Setup datasets and dataloaders
    trainpath = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')
    valpath = os.path.join(os.getcwd(), 'data/val_path_data_labels.csv')

    valpath = '/vol/chameleon/projects/adni/adni_1/val_path_data_labels.csv'
    trainpath = '/vol/chameleon/projects/adni/adni_1/train_path_data_labels.csv'


    trainset = MultiModalDataset(
        path=trainpath, modalities=['tabular'],
        binary_classification=binary_classification,)
    valset = MultiModalDataset(
        path=valpath, modalities=['tabular'],
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
        shuffle=False,
        num_workers=32)

    tabular_model = tabularModel()


    trainer = pl.Trainer(
        max_epochs=hparams['max_epochs'],
        accelerator='gpu',
        devices=1,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=hparams['early_stopping_patience']
            ),
        ]
    )

    trainer.fit(tabular_model, trainloader, valloader)

if __name__ == '__main__':
    hparams = {
        'early_stopping_patience': 5,
        'max_epochs': 1,
        'batch_size': 64,
    }

    train_tabular(hparams)