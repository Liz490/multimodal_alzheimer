import torch
import torch.nn as nn
from pkg.models.pet_models.pet_cnn import Small_PET_CNN
from pkg.models.mri_models.anat_cnn import Anat_CNN

from pkg.loss_functions.focalloss import FocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pkg.models.base_model import Base_Model


class Anat_PET_CNN(Base_Model):

    def __init__(self, hparams, path_pet=None, path_anat=None):
        super().__init__(hparams)

        # load checkpoints
        if path_pet and path_anat:
            self.model_pet = Small_PET_CNN.load_from_checkpoint(path_pet)
            self.model_mri = Anat_CNN.load_from_checkpoint(path_anat)
        else:
            self.model_pet = Small_PET_CNN.load_from_checkpoint(
                hparams["path_pet"])
            self.model_mri = Anat_CNN.load_from_checkpoint(hparams["path_mri"])

        # cut the model after GAP + flatten
        # Note: architectures for 2-class and 3-class
        # problem might deviate slightly
        if hparams["n_classes"] == 2:
            self.model_pet = self.model_pet.model[:-3]
        else:
            self.model_pet = self.model_pet.model[:-1]
        self.model_mri.model.conv_seg = self.model_mri.model.conv_seg[:2]

        # Freeze weights in the stage-1 models
        if ('lr_pretrained' not in hparams.keys()
                or not self.hparams['lr_pretrained']):
            for name, param in self.model_pet.named_parameters():
                param.requires_grad = False
            for name, param in self.model_mri.named_parameters():
                param.requires_grad = False
        # linear layers after concatenation
        self.stage2out = nn.Linear(64 + 64, 64)
        self.cls2 = nn.Linear(64, hparams["n_classes"])

        # non-linearities
        self.relu = nn.ReLU()

        # reduce MRI output to match feature representation of PET output
        self.reduce_dim_mri = nn.Sequential(nn.Linear(512, 64), self.relu)
        # fusion model: takes concatenated outputs of stage 1
        self.model_fuse = nn.Sequential(self.stage2out, self.relu, self.cls2)

        # apply focal loss or weighted cross entropy
        if 'fl_gamma' in hparams and hparams['fl_gamma']:
            self.criterion = FocalLoss(gamma=self.hparams['fl_gamma'])
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=hparams['loss_class_weights'])

    def forward(self, x_pet, x_mri):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        bs = x_mri.shape[0]
        # get outputs of stage-1 models
        out_pet = self.model_pet(x_pet)
        out_mri = self.model_mri(x_mri)
        out_mri = out_mri.view(bs, -1)
        # reduce MRI feature dim
        out_mri = self.reduce_dim_mri(out_mri)

        out = torch.cat((out_pet, out_mri), dim=1)
        out = self.model_fuse(out)
        return out

    def general_step(self, batch, batch_idx, mode):
        x_pet = batch['pet1451']
        x_mri = batch['mri']
        y = batch['label']
        x_pet = x_pet.unsqueeze(1)
        x_mri = x_mri.unsqueeze(1)
        x_pet = x_pet.to(dtype=torch.float32)
        x_mri = x_mri.to(dtype=torch.float32)
        # call the forward method
        y_hat = self(x_pet, x_mri).to(dtype=torch.double)
        loss = self.criterion(y_hat, y)
        self.log(mode + '_loss', loss, on_step=True, prog_bar=True)
        return {'loss': loss, 'outputs': y_hat, 'labels': y}

    def configure_optimizers(self):
        parameters_optim = []
        # if we only want to optimize the parameters of the fusion model
        for name, param in self.model_fuse.named_parameters():
            parameters_optim.append({
                'params': param,
                'lr': self.hparams['lr']})
        for name, param in self.reduce_dim_mri.named_parameters():
            parameters_optim.append({
                'params': param,
                'lr': self.hparams['lr']})
        # if we also want to train stage 1 with a smaller lr
        if self.hparams['lr_pretrained']:
            for name, param in self.model_pet.named_parameters():
                parameters_optim.append({
                    'params': param,
                    'lr': self.hparams['lr_pretrained']})
            for name, param in self.model_mri.named_parameters():
                parameters_optim.append({
                    'params': param,
                    'lr': self.hparams['lr_pretrained']})

        # pass parameters of fusion and reduce-dim network to optimizer
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
