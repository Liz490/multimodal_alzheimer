from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score, MulticlassConfusionMatrix
from pkg.models.tabular_models.dl_approach import load_model
import seaborn as sns
import pandas as pd
import io
import matplotlib.pyplot as plt
from PIL import Image
import torchmetrics
import torchvision
from pkg.models.pet_models.pet_cnn import Small_PET_CNN
from pkg.models.tabular_models.dl_approach import get_avg_activation

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

TRAINPATH='/vol/chameleon/projects/adni/adni_1/train_path_data_labels.csv'
class PET_TABULAR_CNN(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        if hparams["n_classes"] == 3:
            self.label_ind_by_names = {'CN': 0, 'MCI': 1, 'AD': 2}
        else:
            self.label_ind_by_names = {'CN': 0, 'AD': 1}

        # load checkpoints
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
        if not self.hparams['lr_pretrained']:
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

        self.f1_score_train = MulticlassF1Score(num_classes=hparams["n_classes"], average='macro')
        self.f1_score_val = MulticlassF1Score(num_classes=hparams["n_classes"], average='macro')

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
        x_tabular = x_tabular.cpu().squeeze()

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
        x_tabular = batch['tabular']
        y = batch['label']
        x_pet = x_pet.unsqueeze(1)
        x_pet = x_pet.to(dtype=torch.float32)
        # call the forward method
        y_hat = self(x_pet, x_tabular.unsqueeze(1).to(dtype=torch.float32)).to(dtype=torch.double)
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

        # pass parameters of fusion and reduce-dim network to optimizer
        optimizer = torch.optim.Adam(parameters_optim,
                                     weight_decay=self.hparams['l2_reg'])
        # learning rate scheduler
        if self.hparams['reduce_factor_lr_schedule']:
            scheduler = ReduceLROnPlateau(optimizer, factor=self.hparams['reduce_factor_lr_schedule'])
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss_epoch"}
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
        im_train = self.generate_confusion_matrix(training_step_outputs)
        self.logger.experiment.add_image(
            "train_confusion_matrix", im_train, global_step=self.current_epoch)

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
        im_val = self.generate_confusion_matrix(validation_step_outputs)
        self.logger.experiment.add_image(
            "val_confusion_matrix", im_val, global_step=self.current_epoch)

    def generate_confusion_matrix(self, outs):
        """
        See https://stackoverflow.com/a/73388839
        """
        outputs = torch.cat([tmp['outputs'] for tmp in outs])
        labels = torch.cat([tmp['labels'] for tmp in outs])

        task = "binary" if self.hparams['n_classes'] == 2 else "multiclass"
        confusion = torchmetrics.ConfusionMatrix(task=task,
                                                 num_classes=self.hparams["n_classes"]).to(outputs.get_device())
        outputs = torch.max(outputs, dim=1)[1]
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