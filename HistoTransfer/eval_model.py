import os
import copy
import torch
import numpy as np
import pandas as pd
import albumentations
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from albumentations.pytorch import ToTensorV2, ToTensor
from sklearn.metrics import roc_auc_score

from HistoTransfer.dataloader import WSICsvLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def validate_csv_model(df, model, csv_path):
    
    # Put model in eval mode   
    model.eval()
    
    # Get test image dictionary
    test_images = dict(df.groupby('wsi')['path'].apply(list))
    test_images_label = dict(df.groupby('wsi')['label'].apply(max))
    
    test_images_feature = {}
    for k, v in test_images.items():
        test_images_feature[k] = os.path.join(csv_path, k+'.csv')

    test_csv_data = WSICsvLoader(test_images_feature, test_images_label)
    dl = torch.utils.data.DataLoader(test_csv_data, batch_size=1, shuffle=False)    
    
    pred_list = []
    pred_prob_list = []
    label_list = []

    c = 0

    for i, (inputs, labels, _, _) in enumerate(tqdm(dl)):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if inputs.shape[1]<1:
            continue

        with torch.no_grad():
            outputs, _, outputs_attn = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds_prob = F.softmax(outputs, dim=1)[:, 1]
            label_list.append(labels.item())
            pred_list.append(preds.item())
            pred_prob_list.append(preds_prob.item())    

    acc = sum(np.array(pred_list) == np.array(label_list))/len(pred_list)
    auc = roc_auc_score(label_list, pred_prob_list)
    print('Accuracy: {}'.format(sum(np.array(pred_list) == np.array(label_list))/len(pred_list)))
    print('Auc Score: {}'.format(roc_auc_score(label_list, pred_prob_list)))
    
    return acc, auc

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compute_attn_df(tdl, model):
    """ Pass through all the patches in a WSI for validation prediction
    """
    model = copy.deepcopy(model)
    enc_attn = EncAttn(model, head=True)
    enc_attn.eval()
    classifier_layer = model.classifier
    
    attn_rep = torch.tensor([])
    inputs_rep = torch.tensor([])
    fname_list = []

    for i, sample in enumerate(tdl):
        attn_rep_instance, inp_rep_instance = enc_attn(sample[0].cuda())
        attn_rep_instance = attn_rep_instance.detach()
        inp_rep_instance = inp_rep_instance.detach()
        if len(inputs_rep) == 0:
            inputs_rep = inp_rep_instance
            attn_rep = attn_rep_instance
        else:
            inputs_rep = torch.cat((inputs_rep, inp_rep_instance))
            attn_rep = torch.cat((attn_rep, attn_rep_instance))
        fname_list += list(sample[1])

    A = torch.transpose(attn_rep, 1, 0)
    A = F.softmax(A, dim=1)
    M = torch.mm(A, inputs_rep)
    Y_prob = classifier_layer(M)
        
    return torch.max(Y_prob, 1)[1].item(), attn_rep.cpu().numpy().flatten(), F.softmax(Y_prob, dim=1)[:, 1].item()


def eval_test(model, df, data_transforms):
    """ 
    Parameters:
        model - Trained model
        df - dataframe containing following columns:
            1. path - path of patches
            2. wsi - wsi identifier            
            optional - 
            3. label - positive or negative class - if label column 
            is not provided performance is not reported
                
    returns:
        df - dataframe with prediction for each WSI
    """
    
    test_images = dict(df.groupby('wsi')['path'].apply(list))
    pred_list = []
    pred_fname = []
    pred_attn = []
    pred_path = []
    pred_prob = []
    with torch.no_grad():
        for im, im_list in tqdm(test_images.items()):            
            td = WSIPatchLoader(im_list, transform=data_transforms)
            tdl = torch.utils.data.DataLoader(td, batch_size=128,
                                             shuffle=False, num_workers=0)
            t_pred, attn_rep, prob = compute_attn_df(tdl, model)        
            pred_prob += [prob]*len(im_list)
            pred_list += [t_pred]*len(im_list)
            pred_fname += [im]*len(im_list)
            pred_attn += list(attn_rep)
            pred_path += im_list

    pred_df = pd.DataFrame({'wsi': pred_fname, 'prediction': pred_list,\
                            'attention': pred_attn, 'path': pred_path, 'prob': pred_prob})
    
    if 'label' in df.columns:
        test_images_label = dict(df.groupby('wsi')['label'].apply(max))    
        pred_df['actual'] = pred_df['wsi'].apply(lambda x: test_images_label[x])
        pred_wsi_df = pred_df[['wsi', 'prediction', 'actual', 'prob']].drop_duplicates()
        print('Test Accuracy: ', sum(pred_wsi_df['actual']==pred_wsi_df['prediction'])/pred_wsi_df.shape[0])
        print('AUC Score: ', roc_auc_score(pred_df['actual'], pred_df['prob']))
        
    return pred_df
