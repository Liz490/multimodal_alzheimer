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

from MedicalNet.models import resnet
from MedicalNet.model import generate_model
from MedicalNet.setting import parse_opts
import sys

from focalloss import FocalLoss

class IntHandler:
    """
    See https://stackoverflow.com/a/73388839
    """

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
        handlebox.add_artist(text)
        return text


class Anat_CNN(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        if hparams["n_classes"] == 3:
            self.label_ind_by_names = {'CN': 0, 'MCI': 1, 'AD': 2}
        else:
            self.label_ind_by_names = {'CN': 0, 'AD': 1}

        # Initialize Model
        opts = parse_opts()
        opts.pretrain_path = f'/vol/chameleon/projects/adni/adni_1/MedicalNet/pretrain/resnet_{hparams["resnet_depth"]}_23dataset.pth'
        opts.gpu_id = [hparams["gpu_id"]]
        opts.input_W = 91
        opts.input_H = 91
        opts.input_D = 109
        opts.model_depth = hparams["resnet_depth"]

        resnet, _ = generate_model(opts)
        self.model = resnet.module

        modules = nn.ModuleList()

        match hparams["resnet_depth"]:
            case 10:
                n_in = 512
            case 18:
                n_in = 512
            case 50:
                n_in = 2048
            case _:
                raise ValueError("hparams['resnet_depth'] is not in [10, 18, 34, 50]")

        # batchnorm after resnet block or not
        if "batchnorm_begin" in hparams and hparams["batchnorm_begin"]:
            modules.append(nn.BatchNorm3d(n_in))

        # add conv layers if list is not empty
        if 'conv_out' in hparams:
            for n_out, filter_size in zip(self.hparams["conv_out"],
                                        self.hparams["filter_size"]):
                modules.append(nn.Conv3d(n_in, n_out, filter_size, padding='same'))
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

        self.f1_score_train = MulticlassF1Score(num_classes=hparams["n_classes"], average='macro')
        self.f1_score_val = MulticlassF1Score(num_classes=hparams["n_classes"], average='macro')

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        x = self.model(x)

        return x

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
        x = batch['mri']
        y = batch['label']
        x = x.unsqueeze(1)
        x = x.to(dtype=torch.float32)
        y_hat = self.forward(x).to(dtype=torch.double)
        # print(y_hat.shape)
        # print(sys.exit())
        # print(f'ground truth: {y}')
        if mode == 'train':
            # print(f'pred {torch.argmax(y_hat, dim=1)}, gt {y}')
            sftmx = nn.Softmax(dim=1)
            # y_hat_sftmx = sftmx(y_hat)
            # print(f'pred prob {torch.max(y_hat_sftmx, dim=1)}')
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

        return torch.optim.Adam(parameters_optim,
                                weight_decay=self.hparams['l2_reg'])

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

