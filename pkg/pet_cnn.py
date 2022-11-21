import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score, MulticlassConfusionMatrix


class Small_PET_CNN(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = nn.Sequential(
                nn.Conv3d(1, 1, 5, padding='same'),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(1*91*91*109, 3)
                # nn.AvgPool3d(2),
                # nn.Conv3d(32, 32, 5, padding='same'),
                # nn.ReLU(),
                # nn.AvgPool3d(2),
                # nn.Conv3d(32, 32, 5, padding='same'),
                # nn.ReLU(),
                # nn.AvgPool3d(2),
                # nn.Flatten(),
                # nn.
                # maxpool
                # globavg
                # linear
        )
        
        self.criterion = nn.CrossEntropyLoss(weight=hparams['loss_class_weights'])
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
        loss = self.criterion(y_hat, y)
        self.log(mode + '_loss', loss, on_step=True)
        if mode == 'val':
            self.f1_score_val(y_hat, y)
        elif mode == 'train':
            self.f1_score_train(y_hat, y)
        return loss
   
    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
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
