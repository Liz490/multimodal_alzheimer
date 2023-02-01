from pkg.models.tabular_models.dl_approach import *
from pkg.models.mri_models.anat_cnn import Anat_CNN
import torch.nn as nn
from pkg.loss_functions.focalloss import FocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pkg.models.base_model import Base_Model

TRAIN_PATH ='/vol/chameleon/projects/adni/adni_1/train_path_data_labels.csv'


class Tabular_MRT_Model(Base_Model):
    def __init__(self, hparams, path_mri=None):
        super().__init__(hparams)
        # load checkpoints
        if path_mri:
            self.model_mri = Anat_CNN.load_from_checkpoint(path_mri)
        else:
            self.model_mri = Anat_CNN.load_from_checkpoint(hparams["path_mri"])
        self.model_mri.model.conv_seg = self.model_mri.model.conv_seg[:2]

        # Load models and save training sample size
        self.model_tabular, self.tabular_training_size = load_model(TRAINPATH, hparams["n_classes"]==2, ensemble_size=hparams["ensemble_size"])

        # Freeze weights
        if ('lr_pretrained' not in hparams.keys()
                or not self.hparams['lr_pretrained']):
            for name, param in self.model_mri.named_parameters():
                param.requires_grad = False
            for param in self.model_tabular.model[2].parameters():
                param.requires_Grad = False

        # linear layers after concatenation
        self.stage2out = nn.Linear(512 + 512, 64)
        self.cls2 = nn.Linear(64, hparams["n_classes"])

        # non-linearities
        self.relu = nn.ReLU()

        # Reduce output of tabular model
        self.reduce_tab = nn.Sequential(nn.Linear(1024, 512), self.relu)

        self.model_fuse = nn.Sequential(self.stage2out, self.relu, self.cls2)

        if 'fl_gamma' in hparams and hparams['fl_gamma']:
            self.criterion = FocalLoss(gamma=self.hparams['fl_gamma'])
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=hparams['loss_class_weights'])

    def forward(self, x_tabular, x_mri):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        # Forward hook that saves activations at sepcified output layer
        activations = {}
        def get_activations(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        # Register forward hook
        handle = self.model_tabular.model[2].decoder[0].register_forward_hook(get_activations('dec'))
        x_tabular = x_tabular.cpu().squeeze()

        # Forward step for tabular data
        self.model_tabular.predict_proba(x_tabular, normalize_with_test=False)

        # Retrieve activations to use as output for further progressing
        activations = get_avg_activation(activations['dec'], num_ensemble=self.hparams['ensemble_size'] , training_size=self.tabular_training_size)
        handle.remove()

        out_tabular = self.reduce_tab(activations)
        out_mri = self.model_mri(x_mri).squeeze()
        out = torch.cat((out_tabular, out_mri), dim=1)
        out = self.model_fuse(out)
        return out

    def general_step(self, batch, batch_idx, mode):
        x_tabular = batch['tabular']
        x_mri = batch['mri']
        y = batch['label']
        x_mri = x_mri.unsqueeze(1)
        x_mri = x_mri.to(dtype=torch.float32)
        # TODO: double check
        y_hat = self(x_tabular.unsqueeze(1).to(dtype=torch.float32), x_mri).to(dtype=torch.double)

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
            for name, param in self.model_mri.named_parameters():
                parameters_optim.append({
                'params': param,
                'lr': self.hparams['lr_pretrained']})
            for name, param in self.model_tabular.model[2].named_parameters():
                parameters_optim.append({
                    'params': param,
                    'lr': self.hparams['lr_pretrained']})
                
        optimizer = torch.optim.Adam(parameters_optim,
                                weight_decay=self.hparams['l2_reg'])
        if self.hparams['reduce_factor_lr_schedule']:
            scheduler = ReduceLROnPlateau(optimizer, factor=self.hparams['reduce_factor_lr_schedule'])
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss_epoch"}
        else:
            return optimizer
