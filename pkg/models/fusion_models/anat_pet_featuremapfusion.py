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

class IntHandler:
    """
    See https://stackoverflow.com/a/73388839
    """

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
        handlebox.add_artist(text)
        return text


class PET_MRI_FMF(pl.LightningModule):

    def __init__(self, hparams, gpu_id=None):
        super().__init__()
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

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "val")

    def predict_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "pred")

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

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x['loss']
                               for x in training_step_outputs]).mean()
        f1_epoch = self.f1_score_train.compute()
        f1_epoch_per_class = self.f1_score_train_per_class.compute()
        self.f1_score_train.reset()
        self.f1_score_train_per_class.reset()

        log_dict = {
            'train_loss_epoch': avg_loss,
            'train_f1_epoch': f1_epoch,
            'step': float(self.current_epoch)
        }
        for i in range(self.hparams["n_classes"]):
            log_dict[f"train_f1_epoch_class_{i}"] = f1_epoch_per_class[i]
        self.log_dict(log_dict)

        im = self.generate_confusion_matrix(training_step_outputs)
        self.logger.experiment.add_image(
            "train_confusion_matrix", im, global_step=self.current_epoch)

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['loss']
                               for x in validation_step_outputs]).mean()
        f1_epoch = self.f1_score_val.compute()
        f1_epoch_per_class = self.f1_score_val_per_class.compute()
        self.f1_score_val.reset()
        self.f1_score_val_per_class.reset()

        # current_lr = optimizer.param_groups[0]['lr']
        log_dict = {
            'val_loss_epoch': avg_loss,
            'val_f1_epoch': f1_epoch,
            'step': float(self.current_epoch)
        }
        for i in range(self.hparams["n_classes"]):
            log_dict[f"val_f1_epoch_class_{i}"] = f1_epoch_per_class[i]
        self.log_dict(log_dict)

        im = self.generate_confusion_matrix(validation_step_outputs)
        self.logger.experiment.add_image(
            "val_confusion_matrix", im, global_step=self.current_epoch)

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


class Random_Benchmark_All_CN(PET_MRI_FMF):
    def forward(self, x):
        y_hat = torch.zeros_like(super().forward(x))
        y_hat[..., 0] = 1
        y_hat[..., 1:] = 0
        return y_hat
