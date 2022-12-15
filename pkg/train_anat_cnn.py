import os
from numpy import float64
import torch
from torchaudio import transforms
from dataloader import AnatDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from sklearn.metrics import f1_score
from pkg.anat_cnn import Anat_CNN
# from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import sys

def train(hparams):
    pl.seed_everything(15, workers=True)

    # TRANSFORMS
    # transform_train = Compose([
    #     Normalize(mean=hparams['norm_mean_train'], std=hparams['norm_std_train'])
    # ])
    
    # transform_val = Compose([
    #     Normalize(mean=hparams['norm_mean_val'], std=hparams['norm_std_val'])
    # ])
    transform_train = None
    transform_val = None
    
    # DATASET AND DATALOADER
    trainpath = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')
    valpath = os.path.join(os.getcwd(), 'data/val_path_data_labels.csv')

    trainset = AnatDataset(
        path=trainpath, transform=transform_train, normalization=True, binary=True)
    valset = AnatDataset(
        path=valpath, transform=transform_val, normalization=True, binary=True)
    
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

    trainer = pl.Trainer(
        max_epochs=hparams['max_epochs'],
        log_every_n_steps=1,
        accelerator='gpu',
        devices=1,
        callbacks=[EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=hparams['early_stopping_patience']
        )],
        overfit_batches=0.2
    )
    
    trainer.fit(model, trainloader, valloader)


if __name__ == '__main__':
    hparams = {
        'early_stopping_patience': 400,
        'max_epochs': 400,
        'norm_mean_train': 413.6510,
        'norm_std_train': 918.5371,
        'norm_mean_val': 418.4120,
        'norm_std_val': 830.2466,
        'n_classes': 2
    }

    for lr in [1e-4]:
       for bs in [32]:
           hparams['lr'] = lr
           hparams['batch_size'] = bs
           train(hparams)
    # hparams['lr'] = 1e-4
    # hparams['batch_size'] = 32
    # train(hparams)
