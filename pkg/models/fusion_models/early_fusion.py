import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score
import torchvision
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import matplotlib.pyplot as plt
from PIL import Image

from pkg.loss_functions.focalloss import FocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pkg.models.base_model import Base_Model

class PET_MRI_EF(Base_Model):

    def __init__(self, hparams, gpu_id=None):
        super().__init__(hparams, gpu_id=gpu_id)
        self.save_hyperparameters(hparams)
        if hparams["n_classes"] == 3:
            self.label_ind_by_names = {'CN': 0, 'MCI': 1, 'AD': 2}
        else:
            self.label_ind_by_names = {'CN': 0, 'AD': 1}

        modules = nn.ModuleList()

        # Convolutional Block
        # n * (Conv, [Batchnorm], Activation, [Dropout], MaxPool)
        # we need two input channels - one for PET and one for MRI
        n_in = 2
        for n_out, filter_size in zip(self.hparams["conv_out"],
                                      self.hparams["filter_size"]):
            modules.append(nn.Conv3d(n_in, n_out, filter_size, padding="same"))
            if "batchnorm" in self.hparams and self.hparams["batchnorm"]:
                modules.append(nn.BatchNorm3d(n_out))
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool3d(2))
            if "dropout_conv_p" in self.hparams:
                modules.append(nn.Dropout(p=self.hparams["dropout_conv_p"]))
            n_in = n_out

        # Connector Block
        modules.append(nn.AdaptiveAvgPool3d(1))
        modules.append(nn.Flatten())

        # Dense Block
        # n * ([Dropout], Linear)
        if "linear_out" in self.hparams and self.hparams["linear_out"]:
            n_out = self.hparams["linear_out"]
            if "dropout_dense_p" in self.hparams:
                modules.append(
                    nn.Dropout(p=self.hparams["dropout_dense_p"]))
            modules.append(nn.Linear(n_in, n_out))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(n_out, self.hparams["n_classes"]))

        self.model = nn.Sequential(*modules)

        if 'fl_gamma' in hparams and hparams['fl_gamma']:
            self.criterion = FocalLoss(gamma=self.hparams['fl_gamma'])
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=hparams['loss_class_weights'])

        self.criterion = nn.CrossEntropyLoss(
            weight=hparams['loss_class_weights'])
        self.f1_score_train = MulticlassF1Score(
            num_classes=self.hparams["n_classes"], average='macro')
        self.f1_score_train_per_class = MulticlassF1Score(
            num_classes=self.hparams["n_classes"], average='none')
        self.f1_score_val = MulticlassF1Score(
            num_classes=self.hparams["n_classes"], average='macro')
        self.f1_score_val_per_class = MulticlassF1Score(
            num_classes=self.hparams["n_classes"], average='none')

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        return self.model(x)

    def general_step(self, batch, batch_idx, mode):
        x_pet = batch['pet1451']
        x_mri = batch['mri']
        y = batch['label']
        x = torch.stack((x_pet, x_mri), dim=1)
        x = x.to(dtype=torch.float32)
        y_hat = self.forward(x).to(dtype=torch.double)

        loss = self.criterion(y_hat, y)
        if mode != 'pred':
            self.log(mode + '_loss', loss, on_step=True)
        if mode == 'val':
            self.f1_score_val(y_hat, y)
            self.f1_score_val_per_class(y_hat, y)
        elif mode == 'train':
            self.f1_score_train(y_hat, y)
            self.f1_score_train_per_class(y_hat, y)
        return {'loss': loss, 'outputs': y_hat, 'labels': y}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams['lr'])
        if self.hparams['reduce_factor_lr_schedule']:
            scheduler = ReduceLROnPlateau(optimizer, factor=self.hparams['reduce_factor_lr_schedule'])
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss_epoch"}
            #return [optimizer], [scheduler]
        else:
            return optimizer

class Random_Benchmark_All_CN(PET_MRI_EF):
    def forward(self, x):
        y_hat = torch.zeros_like(super().forward(x))
        y_hat[..., 0] = 1
        y_hat[..., 1:] = 0
        return y_hat
