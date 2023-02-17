import torch
import torch.nn as nn

from MedicalNet.model import generate_model
from MedicalNet.setting import parse_opts
import os

from pkg.loss_functions.focalloss import FocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pkg.models.base_model import Base_Model


class Anat_CNN(Base_Model):
    def __init__(self, hparams, gpu_id=None):
        super().__init__(hparams)

        # Initialize Model
        opts = parse_opts()
        opts.pretrain_path = f'/vol/chameleon/projects/adni/adni_1/MedicalNet/pretrain/resnet_{hparams["resnet_depth"]}_23dataset.pth'
        gpu_id = os.getenv('CUDA_VISIBLE_DEVICES')
        if gpu_id:
            opts.gpu_id = gpu_id
        else:
            raise ValueError("CUDA_VISIBLE_DEVICES is not set!")
        opts.input_W = 91
        opts.input_H = 91
        opts.input_D = 109
        opts.model_depth = hparams["resnet_depth"]
        # generate pre-trained resnet
        resnet, _ = generate_model(opts)
        self.model = resnet.module

        # create empty module list that will be filled based on hparams options
        modules = nn.ModuleList()

        # choose resnet depth
        match hparams["resnet_depth"]:
            case 10:
                n_in = 512
            case 18:
                n_in = 512
            case 50:
                n_in = 2048
            case _:
                raise ValueError("hparams['resnet_depth'] is not in \
                    [10, 18, 34, 50]")

        # batchnorm after resnet block or not
        if "batchnorm_begin" in hparams and hparams["batchnorm_begin"]:
            modules.append(nn.BatchNorm3d(n_in))

        # add conv layers if list is not empty
        if 'conv_out' in hparams:
            for n_out, filter_size in zip(self.hparams["conv_out"],
                                          self.hparams["filter_size"]):
                modules.append(nn.Conv3d(n_in, n_out, filter_size,
                                         padding='same'))
                if hparams["batchnorm_conv"]:
                    modules.append(nn.BatchNorm3d(n_out))

                modules.append(nn.ReLU())
                modules.append(nn.MaxPool3d(2))
                n_in = n_out

        # global avg pool
        modules.append(nn.AdaptiveAvgPool3d(1))
        modules.append(nn.Flatten())

        # linear layers
        for n_out in hparams['linear_out']:
            modules.append(nn.Linear(n_in, n_out))
            if "batchnorm_dense" in hparams and hparams["batchnorm_dense"]:
                modules.append(nn.BatchNorm1d(n_out))
            modules.append(nn.ReLU())
            n_in = n_out
        modules.append(nn.Linear(n_in, hparams["n_classes"]))
        modules.append(nn.ReLU())

        self.model.conv_seg = nn.Sequential(*modules)

        if 'fl_gamma' in hparams and hparams['fl_gamma']:
            self.criterion = FocalLoss(gamma=self.hparams['fl_gamma'])
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=hparams['loss_class_weights'])

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        x = self.model(x)

        return x

    def general_step(self, batch, batch_idx, mode):
        x = batch['mri']
        y = batch['label']
        x = x.unsqueeze(1)
        x = x.to(dtype=torch.float32)
        y_hat = self.forward(x).to(dtype=torch.double)

        loss = self.criterion(y_hat, y)
        if mode != 'pred':
            self.log(mode + '_loss', loss, on_step=True, prog_bar=True)
        return {'loss': loss, 'outputs': y_hat, 'labels': y}

    def configure_optimizers(self):
        parameters_optim = []
        for name, param in self.model.named_parameters():
            if 'conv_seg' in name:
                parameters_optim.append({
                    'params': param,
                    'lr': self.hparams['lr']})
            elif 'lr_pretrained' not in self.hparams or not self.hparams['lr_pretrained']:
                param.requires_grad = False
                parameters_optim.append({'params': param})
            else:
                param.requires_grad = True
                parameters_optim.append({
                    'params': param,
                    'lr': self.hparams['lr_pretrained']})

        optimizer = torch.optim.Adam(parameters_optim,
                                     weight_decay=self.hparams['l2_reg'])
        if self.hparams['reduce_factor_lr_schedule']:
            scheduler = ReduceLROnPlateau(
                optimizer, factor=self.hparams['reduce_factor_lr_schedule'])
            return {"optimizer": optimizer,
                    "lr_scheduler": scheduler,
                    "monitor": "val_loss_epoch"}
        else:
            return optimizer
