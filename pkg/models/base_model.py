import torch
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassF1Score
from abc import ABC, abstractmethod
import numpy as np

from pkg.utils.confusion_matrix import generate_loggable_confusion_matrix


class Base_Model(pl.LightningModule, ABC):
    def __init__(self, hparams, gpu_id=None):
        super().__init__()
        self.save_hyperparameters(hparams, ignore=["gpu_id"])

        if hparams["n_classes"] == 3:
            self.label_ind_by_names = {'CN': 0, 'MCI': 1, 'AD': 2}
        else:
            self.label_ind_by_names = {'CN': 0, 'AD': 1}

        self.f1_score_train = MulticlassF1Score(
            num_classes=self.hparams["n_classes"], average='macro')
        self.f1_score_train_per_class = MulticlassF1Score(
            num_classes=self.hparams["n_classes"], average='none')
        self.f1_score_val = MulticlassF1Score(
            num_classes=self.hparams["n_classes"], average='macro')
        self.f1_score_val_per_class = MulticlassF1Score(
            num_classes=self.hparams["n_classes"], average='none')
        self.f1_score_test = MulticlassF1Score(
            num_classes=self.hparams["n_classes"], average='macro')
        self.f1_score_test_per_class = MulticlassF1Score(
            num_classes=self.hparams["n_classes"], average='none')

    @abstractmethod
    def forward(self, x):
        pass

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

    @abstractmethod
    def general_step(self, batch, batch_idx, mode) -> dict:
        pass

    def training_step(self, batch, batch_idx):
        step_output = self.general_step(batch, batch_idx, "train")
        y_hat = step_output["outputs"]
        y = step_output["labels"]
        self.f1_score_train(y_hat, y)
        self.f1_score_train_per_class(y_hat, y)
        return step_output

    def validation_step(self, batch, batch_idx):
        step_output = self.general_step(batch, batch_idx, "val")
        y_hat = step_output["outputs"]
        y = step_output["labels"]
        self.f1_score_val(y_hat, y)
        self.f1_score_val_per_class(y_hat, y)
        return step_output

    def test_step(self, batch, batch_idx):
        step_output = self.general_step(batch, batch_idx, "test")
        y_hat = step_output["outputs"]
        y = step_output["labels"]
        self.f1_score_test(y_hat, y)
        self.f1_score_test_per_class(y_hat, y)
        return step_output

    def predict_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "pred")

    @abstractmethod
    def configure_optimizers(self):
        pass

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

        im = generate_loggable_confusion_matrix(training_step_outputs,
                                                self.label_ind_by_names)
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

        im = generate_loggable_confusion_matrix(validation_step_outputs,
                                                self.label_ind_by_names)
        self.logger.experiment.add_image(
            "val_confusion_matrix", im, global_step=self.current_epoch)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # Bootstrap F1 score
        y_hat = torch.cat([x['outputs'] for x in outputs])
        y_labels = torch.cat([x['labels'] for x in outputs])
        avg_perf, ci = self.f1_score_bootstrap(y_hat, y_labels)

        f1_epoch = self.f1_score_test.compute()
        f1_epoch_per_class = self.f1_score_test_per_class.compute()
        self.f1_score_test.reset()
        self.f1_score_test_per_class.reset()

        log_dict = {
            'test_loss_epoch': avg_loss,
            'test_f1_epoch': f1_epoch,
            'test_f1_epoch_boot': avg_perf,
            'test_f1_epoch_ci': ci,
            'step': float(self.current_epoch)
        }
        for i in range(self.hparams["n_classes"]):
            log_dict[f"test_f1_epoch_class_{i}"] = f1_epoch_per_class[i]
        self.log_dict(log_dict)

        im = generate_loggable_confusion_matrix(outputs,
                                                self.label_ind_by_names)
        self.logger.experiment.add_image(
            "test_confusion_matrix", im, global_step=self.current_epoch)

    def f1_score_bootstrap(self, y_hat, y_labels, n_drawings=1000):
        f1_score_bootstrap = MulticlassF1Score(
            num_classes=self.hparams["n_classes"], average='macro').to(self.device)
        f1_bootstrap = torch.zeros(n_drawings)
        n = len(y_hat)
        for i in range(n_drawings):
            # draw n with replacement from y_hat and y_labels
            random_mask = torch.randint(0, n, (n,))
            y_hat_sample = y_hat[random_mask]
            y_labels_sample = y_labels[random_mask]

            f1_score_bootstrap(y_hat_sample, y_labels_sample)
            f1_bootstrap[i] = f1_score_bootstrap.compute()
            f1_score_bootstrap.reset()

        # here stderr is the standard deviation because
        # f1_bootstrap is not one sample but a sample of samples
        avg_performance = torch.mean(f1_bootstrap)
        stderr = torch.std(f1_bootstrap)
        confidence_interval = 1.96 * stderr

        return avg_performance, confidence_interval
