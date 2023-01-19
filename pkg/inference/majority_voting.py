from operator import mod
import pytorch_lightning as pl
import os
import torch
#from pytorch_lightning.callbacks import ModelCheckpoint
from pkg.models.pet_models.pet_resnet_cnn import PET_CNN_ResNet
from pkg.models.pet_models.pet_cnn import Small_PET_CNN
from pkg.models.mri_models.anat_cnn import Anat_CNN
import sys
from pkg.utils.dataloader import MultiModalDataset
from torch.utils.data import DataLoader

from pkg.models.tabular_models.dl_approach import *
import pkg.models.tabular_models.data_preparation 
import torchmetrics
import pandas as pd
import seaborn as sns
import io
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from pkg.models.pet_models.pet_cnn import IntHandler
from torchmetrics.classification import MulticlassF1Score
'''
different modes: 
see https://www.kaggle.com/code/smsajideen/pytorch-ensemble-analysis-95-26-accuracy
and https://towardsdatascience.com/how-to-attain-a-deep-understanding-of-soft-and-hard-voting-in-ensemble-machine-leaning-methods-32a057fc63fa

1) simply sum up the logits (unweighted soft voting)
2) sum of weighted logits by f1 score (weighted soft voting)
'''

BASEPATH = os.getcwd()
# path dataloader
TRAINPATH = os.path.join(BASEPATH, 'data/train_path_data_labels.csv')
VALPATH = os.path.join(BASEPATH, 'data/val_path_data_labels.csv')
TESTPATH = os.path.join(BASEPATH, 'data/test_path_data_labels.csv')

# path checkpoints
# TODO: update with best model checkpoints
# path for binary classification
# '/u/home/kos/adlm_adni/lightning_logs/optuna_two_class/version_144/checkpoints/epoch=49-step=49.ckpt'
PATH_PET_CNN_2_CLASSES = os.path.join(BASEPATH, 'lightning_logs/best_runs/pet_2_class/checkpoints/epoch=112-step=112.ckpt')
#TODO
PATH_MRI_CNN_2_CLASSES = os.path.join(BASEPATH, 'lightning_logs/best_runs/mri_2_class/checkpoints/epoch=37-step=37.ckpt')
PATH_TABULAR_2_CLASSES = ''
# path for three class classification
PATH_PET_CNN_3_CLASSES = ''
PATH_MRI_CNN_3_CLASSES = ''
PATH_TABULAR_3_CLASSES = ''


# TODO: insert correct values
# F1 scores from training as weights
PET_F1_SCORE_2_CLASS = 0.9673 # train f1: 0.9673, val f1: 0.9005
MRI_F1_SCORE_2_CLASS = 0.922 # train f1: 0.922, val f1: 0.848
TABULAR_F1_SCORE_2_CLASS = 0.9132 # train f1: 0.9132, val f1: 0.8333
PET_F1_SCORE_3_CLASS = 1
MRI_F1_SCORE_3_CLASS = 1
TABULAR_F1_SCORE_3_CLASS = 1

# pass additional gpu_id argument
# this is important because otherwise we load the model on the gpu where this
# specific checkpoint was trained on. If the gpu is now used by another group
# we get a cuda out of memory error
gpu_id = os.getenv('CUDA_VISIBLE_DEVICES')
if not gpu_id:
    raise ValueError('No gpu specified! Please select "export CUDA_VISIBLE_DEVICES=<device_id>')

# transforms
NORMALIZATION_PET = {'mean': 0.5145, 'std': 0.5383}
NORMALIZATION_MRI = {'per_scan_norm': 'min_max'}


# start function here where you pass modalities
def majority_voting(modalities=['pet1451', 't1w', 'tabular'], 
                    binary_classification=True,
                    mode='val'):
    out_dict = {}
    # create a list of models and weights (f1-score of respective model)
    # models = []
    # weights = []
    # to initialize the datasets we first need the quantile information for mri normalization
    quantile = None
    if 't1w' in modalities:
        if binary_classification:
            quantile = Anat_CNN.load_from_checkpoint(PATH_MRI_CNN_2_CLASSES, gpu_id=gpu_id).hparams['norm_percentile']
        else:
            quantile = Anat_CNN.load_from_checkpoint(PATH_MRI_CNN_3_CLASSES, gpu_id=gpu_id).hparams['norm_percentile']

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
    testset = MultiModalDataset(path=TESTPATH,
                            modalities=modalities,
                            normalize_pet=NORMALIZATION_PET,
                            normalize_mri=NORMALIZATION_MRI,
                            quantile=quantile,
                            binary_classification=True)

    trainloader = DataLoader(
            trainset,
            batch_size=len(trainset),
            shuffle=True,
            num_workers=32
        )

    valloader = DataLoader(
            valset,
            ###############################
            batch_size=5, 
            shuffle=False,
            num_workers=32)
    
    valloader_full = DataLoader(
            valset,
            ###############################
            batch_size=len(valset), 
            shuffle=False,
            num_workers=32)

    testloader = DataLoader(
            testset,
            batch_size=len(testset),
            shuffle=False,
            num_workers=32)

    # we need it for the tabular model
    train_batch = next(iter(trainloader))
    assert mode == 'val' or mode == 'test'
    if mode == 'val':
        b = next(iter(valloader_full))
        targets = b['label']
    else:
        b = next(iter(testloader))
        targets = b['label']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # batch = {k:(v.to(device) if k != 'tabular' else v) for k,v in batch.items()}

    logits = []
    weights = []
    logits_pet = None
    logits_mri = None
    logits_tabular = None
    model_mri = None
    for batch in valloader:
        batch = {k:(v.to(device) if k != 'tabular' else v) for k,v in batch.items()}
    
        # PET modality
        if 'pet1451' in modalities:
            if binary_classification:
                #models.append(Small_PET_CNN.load_from_checkpoint(PATH_PET_CNN_2_CLASSES, gpu_id=gpu_id))
                model_pet = Small_PET_CNN.load_from_checkpoint(PATH_PET_CNN_2_CLASSES, gpu_id=gpu_id)
                model_pet.to(device)
                #weights.append(PET_F1_SCORE_2_CLASS)
                weight_pet = PET_F1_SCORE_2_CLASS
            else:
                # models.append(Small_PET_CNN.load_from_checkpoint(PATH_PET_CNN_3_CLASSES, gpu_id=gpu_id))
                model_pet = Small_PET_CNN.load_from_checkpoint(PATH_PET_CNN_3_CLASSES, gpu_id=gpu_id)
                # weights.append(PET_F1_SCORE_3_CLASS)
                weight_pet = PET_F1_SCORE_3_CLASS
            
            model_pet.eval()
            if logits_pet == None:
                logits_pet = model_pet.predict_step(batch=batch,batch_idx=0)['outputs']
                logits_pet = torch.nn.functional.softmax(logits_pet, dim=1)
                
                logits_pet = logits_pet.to('cpu')
                weights.append(weight_pet)
            else:
                logits_pet_i = model_pet.predict_step(batch=batch,batch_idx=0)['outputs']
                logits_pet_i = torch.nn.functional.softmax(logits_pet_i, dim=1)
                
                logits_pet_i = logits_pet_i.to('cpu')
                logits_pet = torch.concat((logits_pet, logits_pet_i))
                logits_pet = logits_pet.to('cpu')

            
            
            

        # MRI modality
        if 't1w' in modalities:
            if binary_classification:
                if model_mri == None:
                    model_mri = Anat_CNN.load_from_checkpoint(PATH_MRI_CNN_2_CLASSES, gpu_id=gpu_id)
                    model_mri.to(device)
                weight_mri = MRI_F1_SCORE_2_CLASS
            else:
                model_mri = Anat_CNN.load_from_checkpoint(PATH_MRI_CNN_3_CLASSES, gpu_id=gpu_id)
                weight_mri = MRI_F1_SCORE_3_CLASS

            model_mri.eval()
            if logits_mri == None:
                logits_mri = model_mri.predict_step(batch=batch,batch_idx=0)['outputs']
                logits_mri = torch.nn.functional.softmax(logits_mri, dim=1)
                
                logits_mri = logits_mri.to('cpu')
                weights.append(weight_mri)

            else:
                logits_mri_i = model_mri.predict_step(batch=batch,batch_idx=0)['outputs']
                logits_mri_i = torch.nn.functional.softmax(logits_mri_i, dim=1)
                
                logits_mri_i = logits_mri_i.to('cpu')
                logits_mri = torch.concat((logits_mri, logits_mri_i))
                logits_mri = logits_mri.to('cpu')
            
            
            
                
        # TABULAR modality
        if 'tabular' in modalities:
            if binary_classification:
                tabular_model = train(train_batch['tabular'], train_batch['label'], binary_classification=True)
                weight_tabular = TABULAR_F1_SCORE_2_CLASS
            else:
                tabular_model = train(train_batch['tabular'], train_batch['label'], binary_classification=False)
                weight_tabular = TABULAR_F1_SCORE_3_CLASS
            if logits_tabular == None:
                _, logits_tabular = predict_batch(batch['tabular'], tabular_model)
                logits_tabular = torch.tensor(logits_tabular)
                
                logits_tabular = logits_tabular.to('cpu')
                weights.append(weight_tabular)
            else:
                _, logits_tabular_i = predict_batch(batch['tabular'], tabular_model)
                logits_tabular_i = torch.tensor(logits_tabular_i)
                logits_tabular = torch.concat((logits_tabular, logits_tabular_i))
                logits_tabular = logits_tabular.to('cpu')

        batch = {k:(v.to('cpu') if k != 'tabular' else v) for k,v in batch.items()}    
            
    # print(logits_pet.shape)        
    if logits_pet != None:
        logits.append(logits_pet)
    if logits_mri != None:
        logits.append(logits_mri)
    if logits_tabular != None:
        logits.append(logits_tabular)


    weights = torch.tensor(weights)

    output_stacked = torch.stack(tuple(logits), dim=2)
    output_weighted = output_stacked * weights.unsqueeze(0).unsqueeze(0)
    # soft voting unweighted: sum of softmax logits over modalities, max over class dimension
    prediction_unweighted = output_stacked.mean(dim=2).argmax(dim=1)
    missclassifications_unweighted = prediction_unweighted != targets
    missclassification_unweighted_idx = torch.nonzero(missclassifications_unweighted)
    
    missclassified_output_unweighted = output_stacked.mean(dim=2)[missclassification_unweighted_idx]
    
    missclassified_output_unweighted = torch.nn.functional.softmax(missclassified_output_unweighted, dim=2)
    missclassified_unweighted_probs, _ = torch.max(missclassified_output_unweighted, dim=2)
    
    missclassified_unweighted_avg_probs = torch.mean(missclassified_unweighted_probs, dim=0)
    
    # soft voting weighted: f1-weighted sum of softmax logits over modalities, max over class dimension
    prediction_weighted = output_weighted.mean(dim=2).argmax(dim=1)
    missclassifications_weighted = prediction_weighted != targets
    missclassification_weighted_idx = torch.nonzero(missclassifications_weighted)
    missclassified_output_weighted = output_weighted.mean(dim=2)[missclassification_weighted_idx]
    missclassified_output_weighted = torch.nn.functional.softmax(missclassified_output_weighted, dim=2)
    missclassified_weighted_probs, _ = torch.max(missclassified_output_weighted, dim=2)
    missclassified_weighted_avg_probs = torch.mean(missclassified_weighted_probs, dim=0)

    out_dict['targets'] = targets
    out_dict['prediction_unweighted'] = prediction_unweighted
    out_dict['prediction_weighted'] = prediction_weighted
    out_dict['logits_unweighted'] = output_stacked.mean(dim=2)
    out_dict['logits_weighted'] = output_weighted.mean(dim=2)
    out_dict['missclassification_unweighted_idx'] = missclassification_unweighted_idx
    out_dict['missclassification_weighted_idx'] = missclassification_weighted_idx
    out_dict['missclassified_output_unweighted'] = missclassified_output_unweighted
    out_dict['missclassified_output_weighted'] = missclassified_output_weighted
    out_dict['missclassified_unweighted_probs'] = missclassified_unweighted_probs
    out_dict['missclassified_unweighted_avg_probs'] = missclassified_unweighted_avg_probs
    out_dict['missclassified_weighted_probs'] = missclassified_weighted_probs
    out_dict['missclassified_weighted_avg_probs'] = missclassified_weighted_avg_probs


    return out_dict

def confusion_matrix(outputs: torch.Tensor,
                     labels: torch.Tensor,
                     binary_classification: bool): #-> plt.figure.Figure:
    if binary_classification:
        label_idx_by_name = {'CN': 0, 'AD': 1}
        n_classes = 2
    else:
        label_idx_by_name = {'CN': 0, 'MCI': 1, 'AD': 2}
        n_classes = 3

    confusion = torchmetrics.ConfusionMatrix(
        num_classes=n_classes)#.to(outputs.get_device())
    confusion(outputs, labels)
    computed_confusion = confusion.compute().numpy().astype(int)#.detach().cpu().numpy().astype(int)

    # confusion matrix
    df_cm = pd.DataFrame(
        computed_confusion,
        index=label_idx_by_name.values(),
        columns=label_idx_by_name.values(),
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.05, right=.65)
    sns.set(font_scale=1.2)
    sns.heatmap(df_cm, annot=True, annot_kws={
                "size": 16}, fmt='d', ax=ax, cmap='crest')
    ax.legend(
        label_idx_by_name.values(),
        label_idx_by_name.keys(),
        handler_map={int: IntHandler()},
        loc='upper left',
        bbox_to_anchor=(1.2, 1)
    )
    return fig
    

if __name__ == "__main__":
    # out_dict['targets'] = targets
    # out_dict['prediction_unweighted'] = prediction_unweighted
    # out_dict['prediction_weighted'] = prediction_weighted
    # out_dict['logits_unweighted'] = output_stacked.mean(dim=2)
    # out_dict['logits_weighted'] = output_weighted.mean(dim=2)
    # out_dict['missclassification_unweighted_idx'] = missclassification_unweighted_idx
    # out_dict['missclassification_weighted_idx'] = missclassification_weighted_idx
    # out_dict['missclassified_output_unweighted'] = missclassified_output_unweighted
    # out_dict['missclassified_output_weighted'] = missclassified_output_weighted
    # out_dict['missclassified_unweighted_probs'] = missclassified_unweighted_probs
    # out_dict['missclassified_unweighted_avg_probs'] = missclassified_unweighted_avg_probs
    # out_dict['missclassified_weighted_probs'] = missclassified_weighted_probs
    # out_dict['missclassified_weighted_avg_probs'] = missclassified_weighted_avg_probs
    
    modalities=['t1w', 'tabular']
    mod_string = ''
    for modal in modalities:
        mod_string += f'_{modal}'

    out = majority_voting(modalities=modalities,
                                    binary_classification=True)
                                
    fig_uw = confusion_matrix(outputs=out['logits_unweighted'], 
                        labels=out['targets'], 
                        binary_classification=True)
    fig_w = confusion_matrix(outputs=out['logits_weighted'], 
                        labels=out['targets'], 
                        binary_classification=True)
    f1_score_uw = MulticlassF1Score(
            num_classes=2, average='macro')
    f1_score_w = MulticlassF1Score(
            num_classes=2, average='macro')
    f1_score_uw(out['logits_unweighted'], out['targets'])
    f1_score_w(out['logits_weighted'], out['targets'])
    f1_uw = f1_score_uw.compute()
    f1_w = f1_score_w.compute()
    print(f'f1 unweighted: {f1_uw}')
    print(f'f1 weighted: {f1_w}')
    idx_uw = out['missclassification_unweighted_idx']
    print(f'missclassified samples unweighted : {idx_uw}')
    idx_w = out['missclassification_weighted_idx']
    print(f'missclassified samples weighted : {idx_w}')
    avg_prob_uw = out['missclassified_unweighted_avg_probs']
    print(f'average prob missclassified samples unweighted : {avg_prob_uw}') 
    avg_prob_w = out['missclassified_weighted_avg_probs']
    print(f'average prob missclassified samples weighted : {avg_prob_w}') 
    fig_uw.savefig(f'pkg/confusion_matrices/unweighted{mod_string}.png')
    fig_w.savefig(f'pkg/confusion_matrices/weighted{mod_string}.png')

    # for a,b in zip(t, pu):
    #     print(a)
    #     print(b)
    #     print('===========')

    #generate_confusion_matrix(targ=t, pred=pu, n_classes=2)

    # #     # TODO
    # #     tabular_model = train(TRAINPATH, binary_classification=True)
    # #     weights.append(TABULAR_F1_SCORE_2_CLASS)

    # # weights = torch.tensor(weights)

    


    


    # # for each modality a list of dicts with 'loss', outputs': tensor of shape batch_size x num_classes , 'labels'
    
    

    

    # #print(batch['tabular'].shape)
    # # print(batch['labels'])
    # # sys.exit()
    # y, logits = predict_batch(batch['tabular'], tabular_model)
    # print(torch.tensor(y.shape))
    # print(torch.tensor(logits).shape)
    # lbls = batch['label']
    # for a,b in zip(lbls, y):
    #     print(a)
    #     print(b)
    #     print('===========')
    # # print(len(batch['pet1451']))
    # # mdl = models[0]
    # # print(mdl.predict_step(batch=batch,batch_idx=0)['outputs'].shape)
    # sys.exit()
    # for model in models:
    #     # for batch in valloader:
    #     #     y_1 = model1(x1)
    #     #     x_tabular = batch['tabular']
    #     model.eval()
    #     predictions.append(trainer.predict(model, valloader))
    # print(predictions)
    # # a list that holds the ensambled prediction for each batch
    # ensamble_predictions = []
    # for prediction_tuple in zip(*predictions):
    #     output_stacked = torch.stack(tuple([sftmx(x['outputs']) for x in prediction_tuple]), dim=2)
    #     output_weighted = output_stacked * weights.unsqueeze(0).unsqueeze(0)
    #     # soft voting unweighted: sum of softmax logits over modalities, max over class dimension
    #     prediction_unweighted = output_stacked.mean(dim=2).argmax(dim=1)
    #     # soft voting weighted: f1-weighted sum of softmax logits over modalities, max over class dimension
    #     prediction_weighted = output_weighted.mean(dim=2).argmax(dim=1)
    #     labels = prediction_tuple[0]['labels']
        
    #     batch_ensamble = {'prediction_unweighted': prediction_unweighted,
    #                     'prediction_weighted': prediction_weighted,
    #                     'labels': labels,
    #                     }
    
    #     ensamble_predictions.append(batch_ensamble)

    # return ensamble_predictions
    


