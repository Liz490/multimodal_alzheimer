# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os.path as osp
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from os import path
from typing import Any, Dict, List, Optional, Type

import torch
import torchvision.transforms as T
from sklearn.model_selection import KFold
from torch.nn import functional as F
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset
from torchmetrics.classification.accuracy import Accuracy

from pytorch_lightning import LightningDataModule, seed_everything, Trainer
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.demos.boring_classes import Net
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.loops.loop import Loop
from pytorch_lightning.trainer.states import TrainerFn

import os
import torch
from dataloader import PETAV1451Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from pkg.models.pet_models.pet_cnn import Small_PET_CNN
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics.classification import MulticlassF1Score

hparams = {
    'early_stopping_patience': 10,
    'max_epochs': 50,
    'norm_mean': 0.5145,
    'norm_std': 0.5383,
    'lr': 0.00075356,
    'batch_size': 64,
    'conv_out': [8, 16, 32, 64],
    'filter_size': [3, 3, 3, 3],
    'batchnorm': False,
    'n_classes': 2,
    'linear_out': 64
}
EXPERIMENT_NAME = 'two_class_kfold'

#############################################################################################
#                           KFold Loop / Cross Validation Example                           #
# This example demonstrates how to leverage Lightning Loop Customization introduced in v1.5 #
# Learn more about the loop structure from the documentation:                               #
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html                  #
#############################################################################################


#############################################################################################
#                           Step 1 / 5: Define KFold DataModule API                         #
# Our KFold DataModule requires to implement the `setup_folds` and `setup_fold_index`       #
# methods.                                                                                  #
#############################################################################################


class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass


#############################################################################################
#                           Step 2 / 5: Implement the KFoldDataModule                       #
# The `KFoldDataModule` will take a train and test dataset.                                 #
# On `setup_folds`, folds will be created depending on the provided argument `num_folds`    #
# Our `setup_fold_index`, the provided train dataset will be split accordingly to        #
# the current fold split.                                                                   #
#############################################################################################


@dataclass
class PETAV1451KFoldDataModule(BaseKFoldDataModule):

    train_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    train_fold: Optional[Dataset] = None
    val_fold: Optional[Dataset] = None

    def prepare_data(self) -> None:
        # our data is already prepared
        pass

    def setup(self, stage: str = None) -> None:
        # TRANSFORMS
        transform = Compose([
            ToTensor(),
            Normalize(mean=hparams['norm_mean'], std=hparams['norm_std'])
        ])

        # DATASET AND DATALOADER
        trainpath = os.path.join(os.getcwd(), 'data/train_path_data_petav1451.csv')
        valpath = os.path.join(os.getcwd(), 'data/val_path_data_petav1451.csv')

        remove_mci = hparams["n_classes"] == 2
        trainset = PETAV1451Dataset(path=trainpath,
                                    transform=transform,
                                    balanced=False,
                                    remove_mci=remove_mci)
        valset = PETAV1451Dataset(path=valpath,
                                transform=transform,
                                balanced=False,
                                remove_mci=remove_mci)
        _, weight_normalized = trainset.get_label_distribution()
        hparams['loss_class_weights'] = 1 - weight_normalized
        self.train_dataset = trainset
        self.test_dataset = valset

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [split for split in KFold(num_folds).split(range(len(self.train_dataset)))]

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.val_fold = Subset(self.train_dataset, val_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_fold, 
                          batch_size=hparams['batch_size'],
                          shuffle=True,
                          num_workers=32)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_fold,
                          batch_size=hparams['batch_size'],
                          num_workers=32)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset)

    def __post_init__(cls):
        super().__init__()


#############################################################################################
#                           Step 3 / 5: Implement the EnsembleVotingModel module            #
# The `EnsembleVotingModel` will take our custom LightningModule and                        #
# several checkpoint_paths.                                                                 #
#                                                                                           #
#############################################################################################


class EnsembleVotingModel(LightningModule):
    def __init__(self, model_cls: Type[LightningModule], checkpoint_paths: List[str]) -> None:
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
        self.test_f1 = MulticlassF1Score(
            num_classes=self.hparams["n_classes"], average='macro')

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Compute the averaged predictions over the `num_folds` models.
        logits = torch.stack([m(batch[0]) for m in self.models]).mean(0)
        loss = F.cross_entropy(logits, batch[1], weight=hparams['loss_class_weights'])
        self.test_f1(logits, batch[1])
        self.log("ensemble_val_f1", self.test_f1)
        self.log("ensemble_val_loss", loss)


#############################################################################################
#                           Step 4 / 5: Implement the  KFoldLoop                            #
# From Lightning v1.5, it is possible to implement your own loop. There is several steps    #
# to do so which are described in detail within the documentation                           #
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html.                 #
# Here, we will implement an outer fit_loop. It means we will implement subclass the        #
# base Loop and wrap the current trainer `fit_loop`.                                        #
#############################################################################################


#############################################################################################
#                     Here is the `Pseudo Code` for the base Loop.                          #
# class Loop:                                                                               #
#                                                                                           #
#   def run(self, ...):                                                                     #
#       self.reset(...)                                                                     #
#       self.on_run_start(...)                                                              #
#                                                                                           #
#        while not self.done:                                                               #
#            self.on_advance_start(...)                                                     #
#            self.advance(...)                                                              #
#            self.on_advance_end(...)                                                       #
#                                                                                           #
#        return self.on_run_end(...)                                                        #
#############################################################################################


class KFoldLoop(Loop):
    def __init__(self, num_folds: int, export_path: str) -> None:
        super().__init__()
        self.num_folds = num_folds
        self.current_fold: int = 0
        self.export_path = export_path

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        print(f"STARTING FOLD {self.current_fold}")
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage.

        # the test loop normally expects the model to be the pure LightningModule, but since we are running the
        # test loop during fitting, we need to temporarily unpack the wrapped module
        wrapped_model = self.trainer.strategy.model
        self.trainer.strategy.model = self.trainer.strategy.lightning_module
        self.trainer.test_loop.run()
        self.trainer.strategy.model = wrapped_model
        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(osp.join(self.export_path, f"model.{self.current_fold}.pt"))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        checkpoint_paths = [osp.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(self.num_folds)]
        voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths)
        voting_model.trainer = self.trainer
        # This requires to connect the new model and move it the right device.
        self.trainer.strategy.connect(voting_model)
        self.trainer.strategy.model_to_device()
        self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)


#############################################################################################
#                           Step 5 / 5: Connect the KFoldLoop to the Trainer                #
# After creating the `KFoldDataModule` and our model, the `KFoldLoop` is being connected to #
# the Trainer.                                                                              #
# Finally, use `trainer.fit` to start the cross validation training.                        #
#############################################################################################

if __name__ == "__main__":
    seed_everything(5)
    # seed_everything(42)
    datamodule = PETAV1451KFoldDataModule()
    datamodule.setup()
    model = Small_PET_CNN(hparams=hparams)
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir='lightning_logs',
        name=EXPERIMENT_NAME)
    trainer = Trainer(
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        num_sanity_val_steps=0,
        max_epochs=hparams['max_epochs'],
        logger=tb_logger,
        log_every_n_steps=1,
        accelerator='gpu',
        devices=1,
        # callbacks=[
        #     EarlyStopping(
        #         monitor='ensemble_val_loss',
        #         mode='min',
        #         patience=hparams['early_stopping_patience'])
        # ]
    )
    internal_fit_loop = trainer.fit_loop
    trainer.fit_loop = KFoldLoop(5, export_path="./kfold_checkpoints/")
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, datamodule)
