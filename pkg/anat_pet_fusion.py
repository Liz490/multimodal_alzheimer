from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score, MulticlassConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import matplotlib.pyplot as plt
from PIL import Image
import torchmetrics
import torchvision
from pet_cnn import Small_PET_CNN
from anat_cnn import Anat_CNN

from MedicalNet.models import resnet
from MedicalNet.model import generate_model
from MedicalNet.setting import parse_opts
import sys

from focalloss import FocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

class IntHandler:
    """
    See https://stackoverflow.com/a/73388839
    """

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
        handlebox.add_artist(text)
        return text


class Anat_PET_CNN(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        if hparams["n_classes"] == 3:
            self.label_ind_by_names = {'CN': 0, 'MCI': 1, 'AD': 2}
        else:
            self.label_ind_by_names = {'CN': 0, 'AD': 1}

        #TODO: if we init new, do we still have the pretrained weights?
        # load PET and MRI checkpoints
        self.model_pet = Small_PET_CNN.load_from_checkpoint(hparams["path_pet"])
        self.model_mri = Anat_CNN.load_from_checkpoint(hparams["path_mri"])
        # load their hparams
        # self.model_pet_hparams = Small_PET_CNN.load_from_checkpoint(hparams["path_pet"]).hparams
        # self.model_mri_hparams = Anat_CNN.load_from_checkpoint(hparams["path_mri"]).hparams
        # keep everything until flatten for PET and MRI
        
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

        # equal contributions
        self.reduce_dim_mri = nn.Sequential(nn.Linear(512, 64), self.relu)
        
        self.model_fuse = nn.Sequential(self.stage2out, self.relu, self.cls2)


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
        out_pet = self.model_pet(x_pet)
        out_mri = self.model_mri(x_mri)
        out_mri = out_mri.view(bs, -1)

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
        # TODO: double check
        y_hat = self(x_pet, x_mri).to(dtype=torch.double)
        
        loss = self.criterion(y_hat, y)
        self.log(mode + '_loss', loss, on_step=True, prog_bar=True)
        if mode == 'val':
            self.f1_score_val(y_hat, y)
        elif mode == 'train':
            self.f1_score_train(y_hat, y)
        return {'loss': loss, 'outputs': y_hat, 'labels': y}

    def training_step(self, batch, batch_idx):
        #pbar = {'train_loss': self.general_step(batch, batch_idx, "train")}
        #return {'loss': self.general_step(batch, batch_idx, "train"), 'progress_bar': pbar}
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
        # TODO: set require_grad=False for everything else
        optimizer = torch.optim.Adam(parameters_optim,
                                weight_decay=self.hparams['l2_reg'])
        if self.hparams['reduce_factor_lr_schedule']:
            scheduler = ReduceLROnPlateau(optimizer, factor=self.hparams['reduce_factor_lr_schedule'])
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss_epoch"}
        else:
            return optimizer

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x['loss']
                               for x in training_step_outputs]).mean()
        f1_epoch = self.f1_score_train.compute()
        self.f1_score_train.reset()

        self.log_dict({
            'train_loss_epoch': avg_loss,
            'train_f1_epoch': f1_epoch,
            'step': float(self.current_epoch)
        })

        im_train = self.generate_confusion_matrix(training_step_outputs)
        self.logger.experiment.add_image(
            "train_confusion_matrix", im_train, global_step=self.current_epoch)

        


    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['loss']
                               for x in validation_step_outputs]).mean()
        f1_epoch = self.f1_score_val.compute()
        self.f1_score_val.reset()

        self.log_dict({
            'val_loss_epoch': avg_loss,
            'val_f1_epoch': f1_epoch,
            'step': float(self.current_epoch)
        })
        im_val = self.generate_confusion_matrix(validation_step_outputs)
        self.logger.experiment.add_image(
            "val_confusion_matrix", im_val, global_step=self.current_epoch)
    
    def generate_confusion_matrix(self, outs):
        """
        See https://stackoverflow.com/a/73388839
        """

        outputs = torch.cat([tmp['outputs'] for tmp in outs])
        labels = torch.cat([tmp['labels'] for tmp in outs])
        confusion = torchmetrics.ConfusionMatrix(
            num_classes=self.hparams["n_classes"]).to(outputs.get_device())
        confusion(outputs, labels)
        computed_confusion = confusion.compute().detach().cpu().numpy().astype(int)

        # confusion matrix
        df_cm = pd.DataFrame(
            computed_confusion,
            index=self.label_ind_by_names.values(),
            columns=self.label_ind_by_names.values(),
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)
        sns.set(font_scale=1.2)
        sns.heatmap(df_cm, annot=True, annot_kws={
                    "size": 16}, fmt='d', ax=ax, cmap='crest')
        ax.legend(
            self.label_ind_by_names.values(),
            self.label_ind_by_names.keys(),
            handler_map={int: IntHandler()},
            loc='upper left',
            bbox_to_anchor=(1.2, 1)
        )
        buf = io.BytesIO()

        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        plt.close('all')
        buf.seek(0)
        with Image.open(buf) as im:
            return torchvision.transforms.ToTensor()(im)



