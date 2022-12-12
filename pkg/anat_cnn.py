from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score, MulticlassConfusionMatrix

from MedicalNet.models import resnet
from MedicalNet.model import generate_model
from MedicalNet.setting import parse_opts
import sys

from focalloss import FocalLoss

class Anat_CNN(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # Initialize Model
        opts = parse_opts()
        opts.pretrain_path = '/vol/chameleon/projects/adni/adni_1/MedicalNet/pretrain/resnet_50_23dataset.pth'
        opts.gpu_id = [0]
        opts.input_W = 91
        opts.input_H = 91
        opts.input_D = 109
        
        resnet, _ = generate_model(opts)
        self.model = resnet.module
        
        self.model.conv_seg = nn.Sequential(
        # nn.Conv3d(2048, 2048, (3, 3, 3), stride=(1, 1, 1), padding='same'),
        #                                 nn.ReLU(),
        #                                 nn.Conv3d(2048, 2048, (3, 3, 3), stride=(1, 1, 1), padding='same'),
        #                                 nn.ReLU(),
        #                                 nn.Conv3d(2048, 2048, (3, 3, 3), stride=(1, 1, 1), padding='same'),
        #                                 nn.ReLU(),
        #                                 nn.MaxPool3d(2),
        #                                 nn.Conv3d(2048, 2048, (3, 3, 3), stride=(1, 1, 1), padding='same'),
        #                                 nn.ReLU(),
        #                                 nn.Conv3d(2048, 2048, (3, 3, 3), stride=(1, 1, 1), padding='same'),
        #                                 nn.ReLU(),
        #                                 nn.Conv3d(2048, 2048, (3, 3, 3), stride=(1, 1, 1), padding='same'),
        #                                 nn.ReLU(),
                                        nn.BatchNorm3d(2048),
                                        nn.AdaptiveAvgPool3d(1),
                                        nn.Flatten(),
                                        nn.Linear(2048, 100),
                                        nn.ReLU(),
                                        nn.Linear(100,3))

        # Only optimize weights in the last few layers
        for name, param in self.model.named_parameters():
            if not 'conv_seg' in name:
                param.requires_grad = False


        
        #for n, l in resnet.module.named_modules:
        # print(self.model)
        # sys.exit()
        # print(self.model.__dict__)
        # print('=======================================================')
        # print('=======================================================')
        # print('=======================================================')
        # self.model = nn.Sequential(
        #     nn.Conv3d(1, 16, 5, padding='same'),
        #     nn.ReLU()
            # nn.MaxPool3d(2),
            # nn.Conv3d(16, 32, 5, padding='same'),
            # nn.ReLU(),
            # nn.MaxPool3d(2),
            # nn.Conv3d(32, 128, 3, padding='same'),
            # nn.ReLU(),
            # nn.MaxPool3d(2),
            # nn.AdaptiveAvgPool3d(1),
            # nn.Flatten(),
            # nn.Linear(128, 3)
        # )
        # print(self.model.__dict__)
        # print(self.model.__dict__)


        self.criterion = nn.CrossEntropyLoss(
            weight=hparams['loss_class_weights'])

        #self.criterion = FocalLoss(gamma=25)
            

        self.f1_score_train = MulticlassF1Score(num_classes=3, average='macro')
        self.f1_score_val = MulticlassF1Score(num_classes=3, average='macro')

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
        (x, y) = batch
        x = x.unsqueeze(1)
        x = x.to(dtype=torch.float32)
        y_hat = self.forward(x).to(dtype=torch.double)
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
        return loss

    def training_step(self, batch, batch_idx):
        #pbar = {'train_loss': self.general_step(batch, batch_idx, "train")}
        #return {'loss': self.general_step(batch, batch_idx, "train"), 'progress_bar': pbar}
        return self.general_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams['lr'])

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

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack(validation_step_outputs).mean()
        f1_epoch = self.f1_score_val.compute()
        self.f1_score_val.reset()

        self.log_dict({
            'val_loss_epoch': avg_loss,
            'val_f1_epoch': f1_epoch,
            'step': float(self.current_epoch)
        })
