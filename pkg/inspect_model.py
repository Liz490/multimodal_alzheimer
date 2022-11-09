
# IMPORTS
import torch
from torch import nn 
from MedicalNet.models import resnet
from MedicalNet.model import generate_model
from MedicalNet.setting import parse_opts
import sys
from dataloader import PETAV1451Dataset
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix
import numpy as np

from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.tensorboard import SummaryWriter
from early_stopping import EarlyStopping
import os
import argparse
from datetime import datetime
import pytz

#PARSED ARGUMENTS
parser1 = argparse.ArgumentParser()

parser1.add_argument("--resume_checkpoint", type=str, default='0', help="checkpoint name")
parser1.add_argument("--device", type=str, default='0', help="gpu 0-7")
parser1.add_argument("--lr", type=float, default=1e-5, help="learning rate")
parser1.add_argument("--epochs", type=int, default=30, help="number of epochs")


args = parser1.parse_args()



# num_of_gpus = torch.cuda.device_count()
# print(num_of_gpus)

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')


earlystop = EarlyStopping(patience=5)



# Initialize Model
opts = parse_opts()
opts.pretrain_path = '/vol/chameleon/projects/adni/adni_1/MedicalNet/pretrain/resnet_50_23dataset.pth'
opts.gpu_id = [args.device]
opts.input_W = 91
opts.input_H = 91
opts.input_D = 109
model, parameters = generate_model(opts)

######################
# TODO: set random seed
######################

transform = Compose([ToTensor(), Normalize(mean=0.5145, std=0.5383)])
path = os.path.join(os.getcwd(), 'data/path_data_petav1451.csv')


# DATASET, SPLIT AND DATALOADER
dataset = PETAV1451Dataset(path=path, transform=transform, balanced=False)  

n_samples_train = int(len(dataset)*0.85)
n_samples_val = len(dataset) - n_samples_train 
trainset, valset= random_split(dataset, [n_samples_train, n_samples_val])
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
valloader = DataLoader(valset, batch_size=32, shuffle=True)



# Loss Function
weight, weight_normalized = dataset.get_label_distribution()
weight = weight.to(dtype=torch.float32).to(device)
weight_normalized = weight_normalized.to(dtype=torch.float32).to(device)
majority_class_freq = torch.max(weight)
weight = majority_class_freq / weight
criterion = nn.CrossEntropyLoss(weight= 1 - weight_normalized)


print(weight)


# Add Layers to finetune the model
# The model is wrapped in a nn.DataParallel object, therefore we cannot access
# model.conv_seg directly, but have to access it over model.module.conv_seg.
model.module.conv_seg = nn.Sequential(nn.Conv3d(2048, 500, (3, 3, 3), stride=(1, 1, 1), padding='same'),
                                    nn.ReLU(),
                                    nn.Conv3d(500, 1, (3, 3, 3), stride=(1, 1, 1), padding='same'),
                                    nn.ReLU(),
                                    nn.Flatten(),
                                    nn.Linear(12*14*12, 100),
                                    nn.ReLU(),
                                    nn.Linear(100,3))

# Load Checkpoint if specified (Default: training from scratch)
epochs_checkpoint = 0
#args.resume_checkpoint = '/vol/chameleon/projects/adni/adni_1/trained_models/pet1451_epoch0.pth'
if args.resume_checkpoint != "0":
    old_checkpoint = torch.load(args.resume_checkpoint)
    epochs_checkpoint = old_checkpoint['epoch']
    model_state_checkpoint = old_checkpoint['model_state']
    model.load_state_dict(model_state_checkpoint)


# Only optimize weights in the last few layers
for name, param in model.named_parameters():
    if not 'conv_seg' in name:
        param.requires_grad = False

# print(torch.rand((4,1,91,109,91)).to(device).get_device())
# #model(torch.rand((4,1,91,109,91)).cuda())
# sys.exit()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)



model = model.to(device) 
running_loss = 0

best_confusion = None
val_loss_last_epoch = 300

writer = SummaryWriter(f'/vol/chameleon/projects/adni/adni_1/tensorboard/runs/lr_{args.lr}')
for epoch in range(args.epochs):
    print(f'EPOCH {epochs_checkpoint + epoch + 1}')
    model.train()
    for i, (x,y) in enumerate(trainloader):
        x = x.unsqueeze(1)
        x = x.to(dtype=torch.float32)
        x = x.to(device)
        y = y.to(device)
        # print(x.device)
        # print(y.device)
        # print(next(model.parameters()).device)
        optimizer.zero_grad()
        pred = model(x)
        
        loss = criterion(pred, y)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if i%10 == 0:
            print(f'Iteration {i+1}:    Train Loss {running_loss/10}')
            writer.add_scalar('training loss', running_loss/10, epoch*len(trainloader)+i)
            # writer.close()
            running_loss = 0
    
    print('VALIDATION')

    val_loss = 0
    model.eval()
    conf_matrix = np.zeros((3,3))
    with torch.no_grad():
        for x, y in valloader:
            x = x.unsqueeze(1)
            x = x.to(dtype=torch.float32)
            x = x.to(device)
            y = y.to(device)
            
            pred = model(x)
            prediction = torch.argmax(pred, 1)
            print(f'pred {prediction}')
            print(f'y {y}')
            # print(y)
            # print(prediction)
            # print(pred)
            # print(y)
            # print(criterion(pred, y).item())
            val_loss += criterion(pred, y).item()
            # print('-')
            # print(y.cpu().view(-1).numpy())
            # print('-')
            # print(prediction.cpu().view(-1).numpy())
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            conf_matrix += confusion_matrix(y.cpu().view(-1).numpy(), prediction.cpu().view(-1).numpy(), labels=[0,1,2])
            # print(conf_matrix)

    

    val_loss /= len(valloader)
    if val_loss < val_loss_last_epoch:
        best_confusion = conf_matrix
    #correct /= n_samples_test
    writer.add_scalar('valdation loss', val_loss, epoch + i)
    writer.close()
    print(f"Avg loss: {val_loss:>8f} \n")
    if earlystop.early_stop(val_loss=val_loss):
        print('STOPPING EARLY')
        epochs = epoch + 1
        break

checkpoint = {
    'epoch': epochs_checkpoint + epoch,
    'model_state': model.state_dict(),
    'confusion': conf_matrix,
    'best_confusion': best_confusion
}

checkpoint_path = '/vol/chameleon/projects/adni/adni_1/trained_models/' \
        + datetime.now(pytz.timezone('Europe/Berlin')).strftime('%Y-%m-%d_%H-%M-%S') \
        + f'_pet1451_epoch{epochs_checkpoint + epoch}.pth'
torch.save(checkpoint, checkpoint_path) 
sys.exit()



a = torch.rand((4,1,91,109,91))

# for a,b in model.named_modules():
#     print(a)



# print(model(a).shape)
# print(model.state_dict().keys())

# print(len(parameters['base_parameters'])) # 159
# print(len(parameters['new_parameters'])) # 8
print(len(model.state_dict()))
for param in model.parameters():
    param.requires_grad = False

print(model.state_dict().keys())



# model.module.conv_seg = nn.Sequential(nn.Linear(2048*12*12*14, 2000*12*12),
#                                     nn.Linear(2000*12*12, 2000*12),
#                                     nn.Linear(2000*12, 100),
#                                     nn.Linear(100, 3))




model = model.cuda() 
print('==================')
print(model.named_parameters)

frozen_params = []
finetune_params = []

for param in model.parameters():
    param.requires_grad = False

sys.exit()
# print(model(a).shape)

# for param in model.parameters():
#     print(param.requires_grad)

torch.save(model, 'pretrained_pet1451.pth')
