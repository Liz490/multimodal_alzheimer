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
import sys

from pkg.loss_functions.focalloss import FocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pkg.models.base_model import Base_Model

class PET_MRI_FMF(Base_Model):

    def __init__(self, hparams, gpu_id=None):
        super().__init__(hparams, gpu_id=gpu_id)
        self.save_hyperparameters(hparams)
        if hparams["n_classes"] == 3:
            self.label_ind_by_names = {'CN': 0, 'MCI': 1, 'AD': 2}
        else:
            self.label_ind_by_names = {'CN': 0, 'AD': 1}
        
        assert hparams['fusion_mode'] == 'concatenate' or hparams['fusion_mode'] == 'maxout'
        self.fusion_mode = hparams['fusion_mode']

        modules_pet = nn.ModuleList()
        modules_mri = nn.ModuleList()
        modules_fused = nn.ModuleList()

        # Convolutional Block
        # n * (Conv, [Batchnorm], Activation, [Dropout], MaxPool)
        # we need two input channels - one for PET and one for MRI

        #PET
        n_in = 1
        for n_out, filter_size in zip(self.hparams["conv_out"],
                                      self.hparams["filter_size"]):
            modules_pet.append(nn.Conv3d(n_in, n_out, filter_size, padding="same"))
            if "batchnorm" in self.hparams and self.hparams["batchnorm"]:
                modules_pet.append(nn.BatchNorm3d(n_out))
            modules_pet.append(nn.ReLU())
            modules_pet.append(nn.MaxPool3d(2))
            if "dropout_conv_p" in self.hparams:
                modules_pet.append(nn.Dropout(p=self.hparams["dropout_conv_p"]))
            n_in = n_out
        # MRI
        n_in = 1
        for n_out, filter_size in zip(self.hparams["conv_out"],
                                      self.hparams["filter_size"]):
            modules_mri.append(nn.Conv3d(n_in, n_out, filter_size, padding="same"))
            if "batchnorm" in self.hparams and self.hparams["batchnorm"]:
                modules_mri.append(nn.BatchNorm3d(n_out))
            modules_mri.append(nn.ReLU())
            modules_mri.append(nn.MaxPool3d(2))
            if "dropout_conv_p" in self.hparams:
                modules_mri.append(nn.Dropout(p=self.hparams["dropout_conv_p"]))
            n_in = n_out

        # two branches with same architecture
        self.backbone_pet = nn.Sequential(*modules_pet)
        self.backbone_mri = nn.Sequential(*modules_mri)

        # model that processes the fused feature map
        if hparams['fusion_mode'] == 'concatenate':
            n_in_fusion = 2*n_in
            #modules_fused.append(nn.Conv3d(n_in, 2*n_in, filter_size, padding="same"))
        else:
            n_in_fusion = n_in
        
        for i in range(hparams["n_layers_fusion"]):
            modules_fused.append(nn.Conv3d(n_in_fusion, hparams["n_out_fusion"], hparams["filter_size_fusion"], padding="same"))
            if "batchnorm_fusion" in self.hparams and self.hparams["batchnorm_fusion"]:
                modules_fused.append(nn.BatchNorm3d(hparams["n_out_fusion"]))
            modules_fused.append(nn.ReLU())
            modules_fused.append(nn.MaxPool3d(2))
            # if "dropout_conv_p" in self.hparams:
            #     modules.append(nn.Dropout(p=self.hparams["dropout_conv_p"]))
            # n_in = n_out
            n_in_fusion = n_in_fusion * 2


        # Connector Block
        modules_fused.append(nn.AdaptiveAvgPool3d(1))
        modules_fused.append(nn.Flatten())

        # Dense Block
        # n * ([Dropout], Linear)
        # if "linear_out" in self.hparams and self.hparams["linear_out"]:
        # n_out = self.hparams["linear_out"]
        if "dropout_dense_p" in self.hparams:
            modules_fused.append(
                nn.Dropout(p=self.hparams["dropout_dense_p"]))
        modules_fused.append(nn.Linear(hparams["n_out_fusion"], 64))
        modules_fused.append(nn.ReLU())
        modules_fused.append(nn.Linear(64, self.hparams["n_classes"]))

        self.fuse_model = nn.Sequential(*modules_fused)

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

    def forward(self, x_pet, x_mri):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        
        out_pet = self.backbone_pet(x_pet)
        out_mri = self.backbone_mri(x_mri)
        if self.fusion_mode == 'concatenate':
            # shape: batchsize x channels x 3d feature cube dims
            out_fused = torch.cat((out_pet, out_mri), dim=1)
        else:
            out_fused = torch.stack((out_pet, out_mri), dim=0)
            out_fused, _ = torch.max(out_fused, dim=0)
        out = self.fuse_model(out_fused)
        return out

    def general_step(self, batch, batch_idx, mode):
        x_pet = batch['pet1451']
        x_mri = batch['mri']
        y = batch['label']
        x_pet = x_pet.unsqueeze(1)
        x_mri = x_mri.unsqueeze(1)
        # x = torch.stack((x_pet, x_mri), dim=1)
        x_pet = x_pet.to(dtype=torch.float32)
        x_mri = x_mri.to(dtype=torch.float32)
        y_hat = self.forward(x_pet=x_pet, x_mri=x_mri).to(dtype=torch.double)

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
        parameters_optim = []
        # if we only want to optimize the parameters of the fusion model
        for name, param in self.backbone_mri.named_parameters():
            parameters_optim.append({
                'params': param,
                'lr': self.hparams['lr']})
        for name, param in self.backbone_pet.named_parameters():
            parameters_optim.append({
                'params': param,
                'lr': self.hparams['lr']})
        for name, param in self.fuse_model.named_parameters():
            parameters_optim.append({
                'params': param,
                'lr': self.hparams['lr']})

        optimizer = torch.optim.Adam(parameters_optim,
                                     weight_decay=self.hparams['l2_reg'])

        if self.hparams['reduce_factor_lr_schedule']:
            scheduler = ReduceLROnPlateau(optimizer, factor=self.hparams['reduce_factor_lr_schedule'])
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss_epoch"}
            #return [optimizer], [scheduler]
        else:
            return optimizer
class Random_Benchmark_All_CN(PET_MRI_FMF):
    def forward(self, x):
        y_hat = torch.zeros_like(super().forward(x))
        y_hat[..., 0] = 1
        y_hat[..., 1:] = 0
        return y_hat
