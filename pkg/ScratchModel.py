# IMPORTS
import torch
from torch import nn
import sys
from dataloader import AnatDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np

from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.tensorboard import SummaryWriter
from early_stopping import EarlyStopping
import os
import argparse
from datetime import datetime
import pytz
from copy import deepcopy
#from ignite.metrics.confusion_matrix import ConfusionMatrix
from torchmetrics import JaccardIndex
from sklearn.metrics import f1_score


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2) -> None:
        super().__init__()
        self.convs = nn.Sequential(*[nn.Conv3d(in_channels, out_channels, 3, padding='same'),
                                     nn.BatchNorm3d(out_channels),
                                     nn.ReLU(),
                                     nn.MaxPool3d(2,stride=stride)])
    def forward(self, x):
        pred = self.convs(x)
        return pred


class simpleCNN(nn.Module):
    def __init__(self, in_channels, outputsize) -> None:
        super().__init__()
        self.c1 = BasicBlock(in_channels=1, out_channels=8)
        self.c2 = BasicBlock(in_channels=8, out_channels=16)
        self.c3 = BasicBlock(in_channels=16, out_channels=32)
        self.c4 = BasicBlock(in_channels=32, out_channels=64)
        self.c5 = BasicBlock(in_channels=64, out_channels=128)
        self.trans = nn.Sequential(*[nn.AdaptiveAvgPool3d(1),
                                     nn.Flatten()])
        self.fc1 = nn.Sequential(*[nn.Linear(outputsize, 1300),
                                   nn.BatchNorm2d(1300),
                                   nn.ReLU()])
        self.fc2 = nn.Sequential(*[nn.Linear(1300, 50),
                                   nn.BatchNorm2d(50),
                                   nn.ReLU()])
        self.fc3 = nn.Linear(50, 3)

    def _slow_forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        x5 = self.c5(x4)
        trans = self.trans(x5)
        f1 = self.fc1(x5)
        f2 = self.fc2(f1)
        preds = self.fc3(f2)
        return preds


def train(lr=3e-4, in_channels=1, outputsize=1300):
    # set the random seed in the very beginning
    torch.manual_seed(5)

    # PARSED ARGUMENTS
    parser1 = argparse.ArgumentParser()

    parser1.add_argument("--resume_checkpoint", type=str, default='0', help="checkpoint name")
    # parser1.add_argument("--device", type=str, default='0', help="gpu 0-7")
    # parser1.add_argument("--lr", type=float, default=1, help="learning rate")
    parser1.add_argument("--epochs", type=int, default=20, help="number of epochs")

    args = parser1.parse_args()
    batch_size = 32
    patience = 5

    # tensorboard
    writer = SummaryWriter(f'/vol/chameleon/projects/adni/adni_1/tensorboard/runs/testmri')

    # Assign device
    # Note: DON'T FORGET TO EXECUTE 'export CUDA_VISIBLE_DEVICES=<device_index>' IN THE TERMINAL
    #       THEN CUDA:0 WILL BE THAT DEVICE
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    earlystop = EarlyStopping(patience=patience)
    stopped_early = False

    # mIoU metric for evaluation
    jaccard = JaccardIndex(num_classes=3).to(device)

    # Initialize Model
    model = simpleCNN(in_channels= in_channels, outputsize= outputsize)

    # TRANSFORMS
    transform = None  # Compose([ToTensor(), Normalize(mean=0.5145, std=0.5383)])  #

    # DATASET AND DATALOADER
    trainpath = os.path.join(os.getcwd(), 'data/train_path_data_labels.csv')
    valpath = os.path.join(os.getcwd(), 'data/val_path_data_labels.csv')

    trainset = AnatDataset(path=trainpath, transform=transform)
    valset = AnatDataset(path=valpath, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=len(valset), shuffle=True)

    print(len(trainset))
    print(len(valset))

    # Loss Function
    weight, weight_normalized = trainset.get_label_distribution()
    weight = weight.to(dtype=torch.float32).to(device)
    weight_normalized = weight_normalized.to(dtype=torch.float32).to(device)
    majority_class_freq = torch.max(weight)
    weight = majority_class_freq / weight
    criterion = nn.CrossEntropyLoss(weight=1 - weight_normalized)  # weight = 1 - weight_normalized

    # Load Checkpoint if specified (Default: training from scratch)
    epochs_checkpoint = 0
    # args.resume_checkpoint = '/vol/chameleon/projects/adni/adni_1/trained_models/2022-11-11_19-44-41_pet1451_epoch4.pth'
    if args.resume_checkpoint != "0":
        print('LOAD CHECKPOINT')
        old_checkpoint = torch.load(args.resume_checkpoint)
        epochs_checkpoint = old_checkpoint['epoch']
        model_state_checkpoint = old_checkpoint['model_state']
        model.load_state_dict(model_state_checkpoint)
    else:
        print('TRAIN FROM SCRATCH')

    # Only optimize weights in the last few layers
    for name, param in model.named_parameters():
        if not 'conv_seg' in name:
            param.requires_grad = False

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initializations
    best_state_dict = deepcopy(model.state_dict())
    best_confusion = None
    running_loss = 0
    miou_last_epoch = 0
    f1_last_epoch = 0
    val_loss_last_epoch = 300

    # send model to device
    model = model.to(device)

    for epoch in range(args.epochs):
        print(f'EPOCH {epochs_checkpoint + epoch}')
        # track loss and acc per epoch
        loss_epoch = 0
        acc_epoch = 0

        # we cannot compute f1 and miou batchwise
        targets_epoch = []
        predictions_epoch = []

        # send model to train mode
        model.train()

        # iterate over train split
        for i, (x, y) in enumerate(trainloader):
            x = x.unsqueeze(1)
            x = x.to(dtype=torch.float32)
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            prediction = torch.argmax(pred, 1)

            # keep track of all predictions and targets that we have in one epoch
            # we can't average over our accuracy measures
            predictions_epoch.append(prediction)
            targets_epoch.append(y)

            loss = criterion(pred, y)
            running_loss += loss.item()
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                if i != 0:
                    print(f'Iteration {i}:    Train Loss {running_loss / 10}')
                    running_loss = 0
        writer.add_scalar('training loss', loss_epoch / len(trainloader), epoch)

        # concatenate all prediction and target tensors into one big tensor
        predictions_epoch = torch.cat(predictions_epoch)
        predictions_epoch = predictions_epoch.cpu().numpy()
        targets_epoch = torch.cat(targets_epoch)
        targets_epoch = targets_epoch.cpu().numpy()

        # compute F1 score
        # 'macro' takes unweighted mean over all classes
        f1_epoch = f1_score(targets_epoch, predictions_epoch, average='macro')
        print(f'Training F1 score over complete epoch: {f1_epoch}')
        writer.add_scalar('training accuracy', f1_epoch, epoch)

        print('VALIDATION')

        val_loss = 0
        model.eval()
        conf_matrix = np.zeros((3, 3))
        with torch.no_grad():
            targets_epoch_val = []
            predictions_epoch_val = []
            for x, y in valloader:
                x = x.unsqueeze(1)
                x = x.to(dtype=torch.float32)
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                prediction = torch.argmax(pred, 1)

                predictions_epoch_val.append(prediction)
                targets_epoch_val.append(y)

                val_loss += criterion(pred, y).item()
                conf_matrix += confusion_matrix(y.cpu().view(-1).numpy(), prediction.cpu().view(-1).numpy(),
                                                labels=[0, 1, 2])

                miou = jaccard(prediction, y).item()

        # This only works if we take the validation batch size as all samples
        val_loss /= len(valloader)
        writer.add_scalar('valdation loss', val_loss, epoch)

        predictions_epoch_val = torch.cat(predictions_epoch_val)
        predictions_epoch_val = predictions_epoch_val.cpu().numpy()
        targets_epoch_val = torch.cat(targets_epoch_val)
        targets_epoch_val = targets_epoch_val.cpu().numpy()

        f1_epoch_val = f1_score(targets_epoch_val, predictions_epoch_val, average='macro')
        print(f'Validation F1 score over complete epoch: {f1_epoch_val}')
        writer.add_scalar('validation accuracy', f1_epoch_val, epoch)

        writer.close()

        if f1_epoch_val > f1_last_epoch:
            best_confusion = conf_matrix.copy()
            best_state_dict = deepcopy(model.state_dict())

            # relevant depending on the criteria for early stopping
            val_loss_last_epoch = val_loss
            miou_last_epoch = miou
            f1_last_epoch = f1_epoch_val

        print(f"Avg loss: {val_loss:>4f} \n")

        if earlystop.early_stop(val_acc=f1_epoch_val):
            print('STOPPING EARLY')
            stopped_early = True
            best_epoch = epoch - patience + 1
            break

    if not stopped_early:
        best_epoch = args.epochs

    checkpoint = {
        'epoch': epochs_checkpoint + best_epoch,
        'model_state': best_state_dict,
        'confusion': best_confusion,
    }

    checkpoint_path = '/vol/chameleon/projects/adni/adni_1/trained_models/' \
                      + datetime.now(pytz.timezone('Europe/Berlin')).strftime('%Y-%m-%d_%H-%M-%S') \
                      + f'_pet1451_epoch{epochs_checkpoint + best_epoch}.pth'
    torch.save(checkpoint, checkpoint_path)


train(lr=1e-4)
sys.exit()



