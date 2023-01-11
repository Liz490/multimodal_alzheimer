from operator import mod
import pytorch_lightning as pl
import os
import torch
#from pytorch_lightning.callbacks import ModelCheckpoint
from pet_resnet_cnn import PET_CNN_ResNet
from pet_cnn import Small_PET_CNN
from anat_cnn import Anat_CNN
import sys
from dataloader import MultiModalDataset
from torch.utils.data import DataLoader

'''
different modes: 
see https://www.kaggle.com/code/smsajideen/pytorch-ensemble-analysis-95-26-accuracy
and https://towardsdatascience.com/how-to-attain-a-deep-understanding-of-soft-and-hard-voting-in-ensemble-machine-leaning-methods-32a057fc63fa

1) simply sum up the logits (unweighted soft voting)
2) sum of weighted logits by f1 score (weighted soft voting)
'''

# init softmax function (we don't want raw logits for soft voting)
sftmx = torch.nn.Softmax(dim=1)

BASEPATH = os.getcwd()
# path dataloader
TRAINPATH = os.path.join(BASEPATH, 'data/train_path_data_labels.csv')
VALPATH = os.path.join(BASEPATH, 'data/val_path_data_labels.csv')

# path checkpoints
# TODO: update with best model checkpoints
# path for binary classification
PATH_PET_CNN_2_CLASSES = os.path.join(BASEPATH, '/u/home/kos/adlm_adni/lightning_logs/optuna_two_class/version_144/checkpoints/epoch=49-step=49.ckpt')
PATH_MRI_CNN_2_CLASSES = os.path.join(BASEPATH, '/u/home/kos/adlm_adni/lightning_logs/optuna_mri_two_class_var_resnet/version_301/checkpoints/epoch=49-step=49.ckpt')
PATH_TABULAR_2_CLASSES = ''
# path for three class classification
PATH_PET_CNN_3_CLASSES = ''
PATH_MRI_CNN_3_CLASSES = ''
PATH_TABULAR_3_CLASSES = ''

# TODO: insert correct values
# F1 scores from training as weights
PET_F1_SCORE_2_CLASS = 1
MRI_F1_SCORE_2_CLASS = 1
TABULAR_F1_SCORE_2_CLASS = 1
PET_F1_SCORE_3_CLASS = 1
MRI_F1_SCORE_3_CLASS = 1
TABULAR_F1_SCORE_3_CLASS = 1

# pass additional gpu_id argument
# this is important because otherwise we load the model on the gpu where this
# specific checkpoint was trained on. If the gpu is now used by another group
# we get a cuda out of memory error
gpu_id = '3'

# transforms
NORMALIZATION_PET = {'mean': 0.5145, 'std': 0.5383}
NORMALIZATION_MRI = {'per_scan_norm': 'min_max'}


# start function here where you pass modalities
def majority_voting(modalities=['pet1451', 't1w', 'tabular'], binary_classification=True):

    # create a list of models and weights (f1-score of respective model)
    models = []
    weights = []
    quantile = None
    # append PET modality
    if 'pet1451' in modalities:
        if binary_classification:
            models.append(Small_PET_CNN.load_from_checkpoint(PATH_PET_CNN_2_CLASSES, gpu_id=gpu_id))
            weights.append(PET_F1_SCORE_2_CLASS)
        else:
            models.append(Small_PET_CNN.load_from_checkpoint(PATH_PET_CNN_3_CLASSES, gpu_id=gpu_id))
            weights.append(PET_F1_SCORE_3_CLASS)
    # append MRI modality
    if 't1w' in modalities:
        if binary_classification:
            models.append(Anat_CNN.load_from_checkpoint(PATH_MRI_CNN_2_CLASSES, gpu_id=gpu_id))
            weights.append(MRI_F1_SCORE_2_CLASS)
            quantile = Anat_CNN.load_from_checkpoint(PATH_MRI_CNN_2_CLASSES, gpu_id=gpu_id).hparams['norm_percentile']
        else:
            models.append(Anat_CNN.load_from_checkpoint(PATH_MRI_CNN_3_CLASSES, gpu_id=gpu_id))
            weights.append(MRI_F1_SCORE_3_CLASS)
            quantile = Anat_CNN.load_from_checkpoint(PATH_MRI_CNN_3_CLASSES, gpu_id=gpu_id).hparams['norm_percentile']
    # append TABULAR modality
    # TODO: ADD TABULAR MODALITY


    weights = torch.tensor(weights)

    # dataset classes
    trainset = MultiModalDataset(path=TRAINPATH, 
                                modalities=modalities,
                                normalize_pet=NORMALIZATION_PET,
                                normalize_mri=NORMALIZATION_MRI,
                                quantile=quantile,
                                binary_classification=True)
    valset = MultiModalDataset(path=VALPATH,
                            modalities=modalities,
                            normalize_pet=NORMALIZATION_PET,
                            normalize_mri=NORMALIZATION_MRI,
                            quantile=quantile,
                            binary_classification=True)

    trainloader = DataLoader(
            trainset,
            batch_size=5,
            shuffle=True,
            num_workers=32
        )

    valloader = DataLoader(
            valset,
            batch_size=5,
            shuffle=False,
            num_workers=32)


    trainer = pl.Trainer()
    predictions = []
    # for each modality a list of dicts with 'loss', outputs': tensor of shape batch_size x num_classes , 'labels'
    for model in models:
        model.eval()
        predictions.append(trainer.predict(model, valloader))

    # a list that holds the ensambled prediction for each batch
    ensamble_predictions = []
    for prediction_tuple in zip(*predictions):
        output_stacked = torch.stack(tuple([sftmx(x['outputs']) for x in prediction_tuple]), dim=2)
        output_weighted = output_stacked * weights.unsqueeze(0).unsqueeze(0)
        # soft voting unweighted: sum of softmax logits over modalities, max over class dimension
        prediction_unweighted = output_stacked.mean(dim=2).argmax(dim=1)
        # soft voting weighted: f1-weighted sum of softmax logits over modalities, max over class dimension
        prediction_weighted = output_weighted.mean(dim=2).argmax(dim=1)
        labels = prediction_tuple[0]['labels']
        
        batch_ensamble = {'prediction_unweighted': prediction_unweighted,
                        'prediction_weighted': prediction_weighted,
                        'labels': labels,
                        }
    
        ensamble_predictions.append(batch_ensamble)

    return ensamble_predictions
    

if __name__ == "__main__":
    ensamble_preds = majority_voting(modalities=['pet1451', 't1w'],
                                    binary_classification=True)
    for ens in ensamble_preds:
        print(ens)
        print("=====================")

