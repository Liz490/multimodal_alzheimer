import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score
from pkg.models.pet_models.pet_cnn import Small_PET_CNN
from pkg.models.mri_models.anat_cnn import Anat_CNN

from pkg.loss_functions.focalloss import FocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pkg.utils.confusion_matrix import generate_loggable_confusion_matrix


class Anat_PET_CNN(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        if hparams["n_classes"] == 3:
            self.label_ind_by_names = {'CN': 0, 'MCI': 1, 'AD': 2}
        else:
            self.label_ind_by_names = {'CN': 0, 'AD': 1}

        # load checkpoints
        self.model_pet = Small_PET_CNN.load_from_checkpoint(hparams["path_pet"])
        self.model_mri = Anat_CNN.load_from_checkpoint(hparams["path_mri"])
        
        # cut the model after GAP + flatten
        self.model_pet = self.model_pet.model[:-3]
        self.model_mri.model.conv_seg = self.model_mri.model.conv_seg[:2]

        # Freeze weights in the stage-1 models
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

        self.f1_score_train = MulticlassF1Score(num_classes=hparams["n_classes"], average='macro')
        self.f1_score_val = MulticlassF1Score(num_classes=hparams["n_classes"], average='macro')
        

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

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

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
        # compute F1
        if mode == 'val':
            self.f1_score_val(y_hat, y)
        elif mode == 'train':
            self.f1_score_train(y_hat, y)
        return {'loss': loss, 'outputs': y_hat, 'labels': y}

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        parameters_optim = []
        # we only want to optimize the parameters of the fusion model
        for name, param in self.model_fuse.named_parameters():
            parameters_optim.append({
                'params': param,
                'lr': self.hparams['lr']})
        for name, param in self.reduce_dim_mri.named_parameters():
            parameters_optim.append({
                'params': param,
                'lr': self.hparams['lr']})
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

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x['loss']
                               for x in training_step_outputs]).mean()
        # F1 score
        f1_epoch = self.f1_score_train.compute()
        self.f1_score_train.reset()
        # additional logging stuff
        self.log_dict({
            'train_loss_epoch': avg_loss,
            'train_f1_epoch': f1_epoch,
            'step': float(self.current_epoch)
        })
        # confusion matrix
        im = generate_loggable_confusion_matrix(training_step_outputs,
                                                self.label_ind_by_names)
        self.logger.experiment.add_image(
            "train_confusion_matrix", im, global_step=self.current_epoch)

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['loss']
                               for x in validation_step_outputs]).mean()
        # F1 score
        f1_epoch = self.f1_score_val.compute()
        self.f1_score_val.reset()
        # additional logging stuff
        self.log_dict({
            'val_loss_epoch': avg_loss,
            'val_f1_epoch': f1_epoch,
            'step': float(self.current_epoch)
        })
        # confusion matrix
        im = generate_loggable_confusion_matrix(validation_step_outputs,
                                                self.label_ind_by_names)
        self.logger.experiment.add_image(
            "val_confusion_matrix", im, global_step=self.current_epoch)
