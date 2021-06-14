import os
import copy
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import albumentations
from albumentations.pytorch import ToTensorV2, ToTensor

from HistoTransfer.dataloader import *
from HistoTransfer.eval_model import *
from HistoTransfer.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_csv_model(model, criterion_dic, optimizer, df, csv_path, alpha=1., beta=0.,
                 num_epochs=25, fpath='checkpoint.pt', verbose=True):
    """ Function for training
    """    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_test_acc = 0.0
        
    # Loss 
    criterion_ce = criterion_dic['CE']
    
    # Separating train, valid images and their labels in a dictionary
    train_images = dict(df.loc[df['is_valid']==0].groupby('wsi')['path'].apply(list))
    valid_images = dict(df.loc[df['is_valid']==1].groupby('wsi')['path'].apply(list))
    train_images_label = dict(df.loc[df['is_valid']==0].groupby('wsi')['label'].apply(max))
    valid_images_label = dict(df.loc[df['is_valid']==1].groupby('wsi')['label'].apply(max))    
    
    train_images_feature = {}
    for k, v in train_images.items():
        train_images_feature[k] = os.path.join(csv_path, k+'.csv')

    valid_images_feature = {}
    for k, v in valid_images.items():
        valid_images_feature[k] = os.path.join(csv_path, k+'.csv')        
        
    train_csv_data = WSICsvLoader(train_images_feature, train_images_label)
    valid_csv_data = WSICsvLoader(valid_images_feature, valid_images_label)

    dataloaders = {'train': torch.utils.data.DataLoader(train_csv_data, batch_size=1, shuffle=True),
                   'val': torch.utils.data.DataLoader(valid_csv_data, batch_size=1, shuffle=True)}
    
    dataset_sizes = {'train': len(train_csv_data), 'val': len(valid_csv_data)}
    
    for epoch in range(num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss_wsi = 0.0
            running_corrects = 0
            if beta != 0:
                running_loss_patch = 0.0

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # Iterate over data.
            for i, (inputs, labels, _, _) in enumerate(dataloaders[phase]):                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, outputs_patch, outputs_attn = model(inputs)                
                    if beta != 0:
                        if labels == 1:
                            patch_labels = torch.ones(len(outputs_patch), dtype=torch.long).to(labels.device)
                        else:
                            patch_labels = torch.zeros(len(outputs_patch), dtype=torch.long).to(labels.device)
                    
                    _, preds = torch.max(outputs, 1)        
                    
                    if beta != 0:
                        loss_patch = criterion_ce(outputs_patch, patch_labels)
                        loss_wsi = criterion_ce(outputs, labels)
                        loss = alpha*loss_wsi + beta*loss_patch
                    else:
                        # Loss
                        loss_wsi = criterion_ce(outputs, labels)
                        loss = alpha*loss_wsi

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # statistics
                running_loss_wsi += loss_wsi.item() * len(inputs)
                running_corrects += torch.sum(preds == labels.data)
                if beta != 0:
                    running_loss_patch += loss_patch.item() * len(inputs)                    
                            
            epoch_loss_wsi = running_loss_wsi / dataset_sizes[phase]
            if beta != 0:
                epoch_loss_patch = running_loss_patch / dataset_sizes[phase]            
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if verbose:
                if beta != 0:
                    print('{} Loss Patch: {:.4f} Loss WSI: {:.4f}  Acc: {:.4f}'.format(
                        phase, epoch_loss_patch, epoch_loss_wsi, epoch_acc))
                else:
                    print('{} Loss WSI: {:.4f}  Acc: {:.4f}'.format(
                        phase, epoch_loss_wsi, epoch_acc))
            
            if phase == 'val':
                if epoch_acc >= best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    checkpoint = {
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    save_ckp(checkpoint, fpath)                               


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Final epoch model
    model_final = copy.deepcopy(model)
    
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model    

def finetune_model(model, criterion_dic, optimizer, df, data_transforms, alpha=1., beta=0., num_epochs=25, fpath='checkpoint.pt'):
    """ Function for training
    """    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_test_acc = 0.0
    best_loss = 100
        
    # Loss 
    criterion_ce = criterion_dic['CE']
    
    # Separating train, valid images and their labels in a dictionary
    train_images = dict(df.loc[df['is_valid']==0].groupby('wsi')['path'].apply(list))
    valid_images = dict(df.loc[df['is_valid']==1].groupby('wsi')['path'].apply(list))
    train_images_label = dict(df.loc[df['is_valid']==0].groupby('wsi')['label'].apply(max))
    valid_images_label = dict(df.loc[df['is_valid']==1].groupby('wsi')['label'].apply(max))    

    train_data = WSIDataloader(train_images, train_images_label, transform=data_transforms)
    val_data = WSIDataloader(valid_images, valid_images_label, transform=data_transforms)

    batch_size = 1
    num_workers = 0

    dataloaders = {'train': torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers),
                  'val': torch.utils.data.DataLoader(val_data, batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers)}
    dataset_sizes = {'train': len(train_data), 'val': len(val_data)}    
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                model.apply(set_bn_eval)
            else:
                model.eval()   # Set model to evaluate mode

            running_loss_wsi = 0.0
            running_corrects = 0
            if beta != 0:
                running_loss_patch = 0.0

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # Iterate over data.
            for i, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, outputs_patch, outputs_attn = model(inputs)                
                    if beta != 0:
                        if labels == 1:
                            patch_labels = torch.ones(len(outputs_patch), dtype=torch.long).to(labels.device)
                        else:
                            patch_labels = torch.zeros(len(outputs_patch), dtype=torch.long).to(labels.device)
                    
                    _, preds = torch.max(outputs, 1)        
                    
                    if beta != 0:
                        loss_patch = criterion_ce(outputs_patch, patch_labels)
                        loss_wsi = criterion_ce(outputs, labels)
                        loss = alpha*loss_wsi + beta*loss_patch
                    else:
                        # Loss
                        loss_wsi = criterion_ce(outputs, labels)
                        loss = alpha*loss_wsi
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # statistics
                running_loss_wsi += loss_wsi.item() * len(inputs)
                running_corrects += torch.sum(preds == labels.data)
                if beta != 0:
                    running_loss_patch += loss_patch.item() * len(inputs)                    
                
            epoch_loss_wsi = running_loss_wsi / dataset_sizes[phase]
            if beta != 0:
                epoch_loss_patch = running_loss_patch / dataset_sizes[phase]            
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if beta != 0:
                print('{} Loss Patch: {:.4f} Loss WSI: {:.4f}  Acc: {:.4f}'.format(
                    phase, epoch_loss_patch, epoch_loss_wsi, epoch_acc))
            else:
                print('{} Loss WSI: {:.4f}  Acc: {:.4f}'.format(
                    phase, epoch_loss_wsi, epoch_acc))
            
            if phase == 'val':
#                 if epoch_loss_wsi < best_loss:
#                     best_loss = epoch_loss_wsi
#                     best_acc = epoch_acc
#                     best_model_wts = copy.deepcopy(model.state_dict())
#                     checkpoint = {
#                         'state_dict': model.state_dict(),
#                         'optimizer': optimizer.state_dict()
#                     }
#                     save_ckp(checkpoint, fpath)                               
                
                if epoch_acc >= best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    checkpoint = {
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    save_ckp(checkpoint, fpath)                               


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Final epoch model
    model_final = copy.deepcopy(model)
    
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model    