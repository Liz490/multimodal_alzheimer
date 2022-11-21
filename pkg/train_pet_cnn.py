import os
import torch
from dataloader import PETAV1451Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from sklearn.metrics import f1_score
from pet_cnn import Small_PET_CNN
# from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

if __name__ == '__main__':
    pl.seed_everything(5, workers=True)

    hparams = {
        'batch_size': 32,
        'early_stopping_patience': 5,
        'max_epochs': 20,
        'norm_mean': 0.5145,
        'norm_std': 0.5383
    }

    # TRANSFORMS
    transform = Compose([
        ToTensor(),
        Normalize(mean=hparams['norm_mean'], std=hparams['norm_std'])
    ])

    # DATASET AND DATALOADER
    trainpath = os.path.join(os.getcwd(), 'data/train_path_data_petav1451.csv')
    valpath = os.path.join(os.getcwd(), 'data/val_path_data_petav1451.csv')

    trainset = PETAV1451Dataset(path=trainpath, transform=transform, balanced=False)
    valset = PETAV1451Dataset(path=valpath, transform=transform, balanced=False)

    trainloader = DataLoader(trainset, batch_size=hparams['batch_size'], shuffle=True, num_workers=32)
    valloader = DataLoader(valset, batch_size=len(valset), num_workers=32)
 
    _, weight_normalized = trainset.get_label_distribution()
    hparams['loss_class_weights'] = 1 - weight_normalized

    model = Small_PET_CNN(hparams=hparams)

    trainer = pl.Trainer(
        max_epochs=hparams['max_epochs'],
        log_every_n_steps=10,
        accelerator='gpu',
        devices=1,
        callbacks=[EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=hparams['early_stopping_patience']
        )]
    )

    trainer.fit(model, trainloader, valloader)


        
