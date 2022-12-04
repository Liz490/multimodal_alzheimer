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


class IntHandler:
    """
    See https://stackoverflow.com/a/73388839
    """

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
        handlebox.add_artist(text)
        return text


class Small_PET_CNN(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.n_classes = 3
        self._label_ind_by_names = {'CN': 0, 'MCI': 1, 'AD': 2}

        self.model = nn.Sequential(
            nn.Conv3d(1, 16, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 5, padding='same'),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 128, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(128, 3)
        )

        self.criterion = nn.CrossEntropyLoss(
            weight=hparams['loss_class_weights'])
        self.f1_score_train = MulticlassF1Score(
            num_classes=self.n_classes, average='macro')
        self.f1_score_train_per_class = MulticlassF1Score(
            num_classes=self.n_classes, average='none')
        self.f1_score_val = MulticlassF1Score(
            num_classes=self.n_classes, average='macro')
        self.f1_score_val_per_class = MulticlassF1Score(
            num_classes=self.n_classes, average='none')

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
            self.f1_score_val_per_class(y_hat, y)
        elif mode == 'train':
            self.f1_score_train(y_hat, y)
            self.f1_score_train_per_class(y_hat, y)
        return {'loss': loss, 'outputs': y_hat, 'labels': y}

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams['lr'])

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x['loss']
                               for x in training_step_outputs]).mean()
        f1_epoch = self.f1_score_train.compute()
        f1_epoch_per_class = self.f1_score_train_per_class.compute()
        self.f1_score_train.reset()
        self.f1_score_train_per_class.reset()

        self.log_dict({
            'train_loss_epoch': avg_loss,
            'train_f1_epoch': f1_epoch,
            'train_f1_epoch_class_0': f1_epoch_per_class[0],
            'train_f1_epoch_class_1': f1_epoch_per_class[1],
            'train_f1_epoch_class_2': f1_epoch_per_class[2],
            'step': float(self.current_epoch)
        })

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

        self.log_dict({
            'val_loss_epoch': avg_loss,
            'val_f1_epoch': f1_epoch,
            'val_f1_epoch_class_0': f1_epoch_per_class[0],
            'val_f1_epoch_class_1': f1_epoch_per_class[1],
            'val_f1_epoch_class_2': f1_epoch_per_class[2],
            'step': float(self.current_epoch)
        })

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
            num_classes=self.n_classes).to(outputs.get_device())
        confusion(outputs, labels)
        computed_confusion = confusion.compute().detach().cpu().numpy().astype(int)

        # confusion matrix
        df_cm = pd.DataFrame(
            computed_confusion,
            index=self._label_ind_by_names.values(),
            columns=self._label_ind_by_names.values(),
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)
        sns.set(font_scale=1.2)
        sns.heatmap(df_cm, annot=True, annot_kws={
                    "size": 16}, fmt='d', ax=ax, cmap='crest')
        ax.legend(
            self._label_ind_by_names.values(),
            self._label_ind_by_names.keys(),
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

