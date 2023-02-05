import torch
import torch.nn as nn
from pkg.models.tabular_models.dl_approach import load_model
from pkg.models.pet_models.pet_cnn import Small_PET_CNN
from pkg.models.tabular_models.dl_approach import get_avg_activation

from pkg.loss_functions.focalloss import FocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pkg.models.base_model import Base_Model


TRAINPATH = '/vol/chameleon/projects/adni/adni_1/train_path_data_labels.csv'


class PET_TABULAR_CNN(Base_Model):

    def __init__(self, hparams, path_pet=None):
        super().__init__(hparams)

        # load checkpoints
        if path_pet:
            self.model_pet = Small_PET_CNN.load_from_checkpoint(path_pet)
        else:
            self.model_pet = Small_PET_CNN.load_from_checkpoint(hparams["path_pet"])

        # cut the model after GAP + flatten
        # Note: architectures for 2-class and 3-class problem might deviate slightly
        if hparams["n_classes"] == 2:
            self.model_pet = self.model_pet.model[:-3]
        else:
            self.model_pet = self.model_pet.model[:-1]

        # Load models and save training sample size
        self.model_tabular, self.tabular_training_size = load_model(TRAINPATH, hparams["n_classes"] == 2,
                                                                    ensemble_size=hparams["ensemble_size"])

        # Freeze weights in the stage-1 models
        if ('lr_pretrained' not in hparams.keys()
                or not self.hparams['lr_pretrained']):
            for name, param in self.model_pet.named_parameters():
                param.requires_grad = False
            for param in self.model_tabular.model[2].parameters():
                param.requires_Grad = False


        # linear layers after concatenation
        self.stage2out = nn.Linear(64 + 64, 64)
        self.cls2 = nn.Linear(64, hparams["n_classes"])

        # non-linearities
        self.relu = nn.ReLU()

        # reduce Tabular output to match feature representation of PET output
        if self.hparams['simple_dim_red']:
            self.reduce_tab = nn.Sequential(nn.Linear(1024, 512), self.relu, nn.Linear(512,64), self.relu)
        else:
            self.reduce_tab = nn.Sequential(nn.Linear(1024, 64), self.relu)

        # fusion model: takes concatenated outputs of stage 1
        self.model_fuse = nn.Sequential(self.stage2out, self.relu, self.cls2)

        # apply focal loss or weighted cross entropy
        if 'fl_gamma' in hparams and hparams['fl_gamma']:
            self.criterion = FocalLoss(gamma=self.hparams['fl_gamma'])
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=hparams['loss_class_weights'])

    def forward(self, x_pet, x_tabular):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        # get outputs of stage-1 models
        out_pet = self.model_pet(x_pet)

        activations = {}
        def get_activations(name):
            def hook(model, input, output):
                activations[name] = output.detach()

            return hook

        # Register forward hook
        handle = self.model_tabular.model[2].decoder[0].register_forward_hook(get_activations('dec'))
        x_tabular = x_tabular.cpu().squeeze(dim=1)

        # Forward step for tabular data
        self.model_tabular.predict_proba(x_tabular, normalize_with_test=False)

        # Retrieve activations to use as output for further progressing
        activations = get_avg_activation(activations['dec'], num_ensemble=self.hparams['ensemble_size'],
                                         training_size=self.tabular_training_size)
        handle.remove()

        # reduce Tabular feature dim
        out_tab = self.reduce_tab(activations)

        out = torch.cat((out_pet, out_tab), dim=1)
        out = self.model_fuse(out)
        return out

    def general_step(self, batch, batch_idx, mode):
        x_pet = batch['pet1451']
        x_tabular = batch['tabular']
        y = batch['label']
        x_pet = x_pet.unsqueeze(1)
        x_pet = x_pet.to(dtype=torch.float32)
        # call the forward method
        y_hat = self(x_pet, x_tabular.unsqueeze(1).to(dtype=torch.float32)).to(dtype=torch.double)
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
        for name, param in self.reduce_tab.named_parameters():
            parameters_optim.append({
                'params': param,
                'lr': self.hparams['lr']})

        # if we also want to train stage 1 with a smaller lr
        if self.hparams['lr_pretrained']:
            for name, param in self.model_pet.named_parameters():
                parameters_optim.append({
                'params': param,
                'lr': self.hparams['lr_pretrained']})
            for name, param in self.model_tabular.model[2].named_parameters():
                parameters_optim.append({
                    'params': param,
                    'lr': self.hparams['lr_pretrained']})

        # pass parameters of fusion and reduce-dim network to optimizer
        optimizer = torch.optim.Adam(parameters_optim,
                                     weight_decay=self.hparams['l2_reg'])
        # learning rate scheduler
        if self.hparams['reduce_factor_lr_schedule']:
            scheduler = ReduceLROnPlateau(optimizer, factor=self.hparams['reduce_factor_lr_schedule'])
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss_epoch"}
        else:
            return optimizer
