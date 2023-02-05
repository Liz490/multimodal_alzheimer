import torch
import torch.nn as nn
from pkg.models.fusion_models.anat_pet_fusion import Anat_PET_CNN
from pkg.models.fusion_models.tabular_mri_fusion import Tabular_MRT_Model
from pkg.models.fusion_models.pet_tabular_fusion import PET_TABULAR_CNN

from pkg.loss_functions.focalloss import FocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pkg.models.base_model import Base_Model


class All_Modalities_Fusion(Base_Model):
    def __init__(self, hparams):
        super().__init__(hparams)

        # load checkpoints
        self.model_anat_pet = Anat_PET_CNN.load_from_checkpoint(
            hparams["path_anat_pet"],
            path_pet=hparams['path_pet'],
            path_anat=hparams['path_anat'])
        self.model_anat_tab = Tabular_MRT_Model.load_from_checkpoint(
            hparams["path_anat_tab"],
            path_mri=hparams['path_anat'])
        self.model_pet_tab = PET_TABULAR_CNN.load_from_checkpoint(
            hparams["path_pet_tab"],
            path_pet=hparams['path_pet'])

        # cut of the classifiers from the second stage models
        self.model_anat_pet.model_fuse = self.model_anat_pet.model_fuse[:-2]
        self.model_anat_tab.model_fuse = self.model_anat_tab.model_fuse[:-2]
        self.model_pet_tab.model_fuse = self.model_pet_tab.model_fuse[:-2]

        # Freeze weights in the stage-2 models
        if ('lr_pretrained' not in hparams.keys()
                or not self.hparams['lr_pretrained']):
            for _, param in self.model_anat_pet.reduce_dim_mri.named_parameters():
                param.requires_grad = False
            for _, param in self.model_anat_pet.model_fuse.named_parameters():
                param.requires_grad = False
            for _, param in self.model_anat_tab.reduce_tab.named_parameters():
                param.requires_grad = False
            for _, param in self.model_anat_tab.model_fuse.named_parameters():
                param.requires_grad = False
            for _, param in self.model_pet_tab.model_fuse.named_parameters():
                param.requires_grad = False
            for _, param in self.model_pet_tab.reduce_tab.named_parameters():
                param.requires_grad = False

        # linear layers after concatenation
        self.stage3out = nn.Linear(64 + 64 + 64, 64)
        self.cls3 = nn.Linear(64, hparams["n_classes"])

        # non-linearities
        self.relu = nn.ReLU()

        # fusion model: takes concatenated outputs of stage 2
        self.model_fuse = nn.Sequential(self.stage3out, self.relu, self.cls3)

        # apply focal loss or weighted cross entropy
        if 'fl_gamma' in hparams and hparams['fl_gamma']:
            self.criterion = FocalLoss(gamma=self.hparams['fl_gamma'])
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=hparams['loss_class_weights'])

    def forward(self, x_pet, x_mri, x_tab):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        out_anat_pet = self.model_anat_pet(x_pet, x_mri)
        out_anat_tab = self.model_anat_tab(x_tab, x_mri)
        out_pet_tab = self.model_pet_tab(x_pet, x_tab)
        out = torch.cat((out_anat_pet, out_anat_tab, out_pet_tab), dim=1)
        out = self.model_fuse(out)
        return out

    def general_step(self, batch, batch_idx, mode):
        x_pet = batch['pet1451']
        x_mri = batch['mri']
        x_tab = batch['tabular']
        y = batch['label']
        x_pet = x_pet.unsqueeze(1)
        x_mri = x_mri.unsqueeze(1)
        x_tab = x_tab.unsqueeze(1)
        x_pet = x_pet.to(dtype=torch.float32)
        x_mri = x_mri.to(dtype=torch.float32)
        x_tab = x_tab.to(dtype=torch.float32)
        # call the forward method
        y_hat = self(x_pet, x_mri, x_tab).to(dtype=torch.double)
        loss = self.criterion(y_hat, y)
        self.log(mode + '_loss', loss, on_step=True, prog_bar=True)
        return {'loss': loss, 'outputs': y_hat, 'labels': y}

    def configure_optimizers(self):
        parameters_optim = []
        # we only want to optimize the parameters of the fusion model
        for _, param in self.model_fuse.named_parameters():
            parameters_optim.append({
                'params': param,
                'lr': self.hparams['lr']})

        if self.hparams['lr_pretrained']:
            previous_stage_models = [self.model_anat_pet.model_pet,
                                     self.model_anat_pet.model_mri,
                                     self.model_anat_pet.stage2out,
                                     self.model_anat_pet.reduce_dim_mri,
                                     self.model_pet_tab.model_pet,
                                     self.model_pet_tab.model_tabular,
                                     self.model_pet_tab.stage2out,
                                     self.model_pet_tab.reduce_tab,
                                     self.model_anat_tab.model_mri,
                                     self.model_anat_tab.model_tabular,
                                     self.model_anat_tab.stage2out,
                                     self.model_anat_tab.reduce_tab
                                     ]
            # unfreeze all previous stages
            for model in previous_stage_models:
                for _, param in model.named_parameters():
                    parameters_optim.append({
                        'params': param,
                        'lr': self.hparams['lr_pretrained']})
        # pass parameters of fusion network to optimizer
        optimizer = torch.optim.Adam(parameters_optim,
                                     weight_decay=self.hparams['l2_reg'])
        # learning rate scheduler
        if self.hparams['reduce_factor_lr_schedule']:
            scheduler = ReduceLROnPlateau(
                optimizer, factor=self.hparams['reduce_factor_lr_schedule'])
            return {"optimizer": optimizer,
                    "lr_scheduler": scheduler,
                    "monitor": "val_loss_epoch"}
        else:
            return optimizer
