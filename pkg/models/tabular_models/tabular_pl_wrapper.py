from pkg.models.tabular_models.dl_approach import *
from pkg.models.mri_models.anat_cnn import Anat_CNN
import torch.nn as nn
from pkg.loss_functions.focalloss import FocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pkg.models.base_model import Base_Model

TRAIN_PATH = '/vol/chameleon/projects/adni/adni_1/train_path_data_labels.csv'
NOT_IMPLEMENTED_ERROR_MESSAGE = (
    "The tabular model was not trained with pl." +
    "This wrapper is just for testing purposes."
)


class Tabular_Model(Base_Model):
    def __init__(self, hparams):
        super().__init__(hparams)

        binary = (hparams["n_classes"] == 2)

        # Load models and save training sample size
        self.model_tabular, self.tabular_training_size = load_model(
            TRAINPATH,
            binary,
            ensemble_size=hparams["ensemble_size"]
        )

        self.criterion = nn.CrossEntropyLoss(
            weight=hparams['loss_class_weights'])

    def forward(self, x_tabular):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        x_tabular = x_tabular.cpu().squeeze(dim=1)

        # Forward step for tabular data
        out = self.model_tabular.predict_proba(
            x_tabular, normalize_with_test=False)
        out = torch.tensor(out, device=self.device)

        return out

    def general_step(self, batch, batch_idx, mode):
        if mode == 'train':
            raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)
        if mode == 'val':
            raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)
        x_tabular = batch['tabular']
        y = batch['label']
        # TODO: double check
        x_tabular = x_tabular.unsqueeze(1).to(dtype=torch.float32)
        y_hat = self(x_tabular).to(dtype=torch.double)

        loss = self.criterion(y_hat, y)
        self.log(mode + '_loss', loss, on_step=True, prog_bar=True)
        return {'loss': loss, 'outputs': y_hat, 'labels': y}

    def configure_optimizers(self):
        raise NotImplementedError(NOT_IMPLEMENTED_ERROR_MESSAGE)
        # parameters_optim = []

        # if self.hparams['lr']:
        #     for _, param in self.model_tabular.model[2].named_parameters():
        #         parameters_optim.append({
        #             'params': param,
        #             'lr': self.hparams['lr']})

        # optimizer = torch.optim.Adam(
        #     parameters_optim, weight_decay=self.hparams['l2_reg'])

        # if self.hparams['reduce_factor_lr_schedule']:
        #     scheduler = ReduceLROnPlateau(
        #         optimizer, factor=self.hparams['reduce_factor_lr_schedule'])
        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": scheduler,
        #         "monitor": "val_loss_epoch"}
        # else:
        #     return optimizer
