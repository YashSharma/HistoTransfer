import os
import torch
import numpy as np
import pandas as pd

from HistoTransfer.dataloader import WSICsvLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()

def save_ckp(state, fpath=None):
    ''' Save model
    '''
    if fpath == None:
        fpath =  'checkpoint.pt'
    torch.save(state, fpath)
    
def load_ckp(checkpoint_fpath, model, optimizer):
    ''' load model
    '''
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def get_attn_patches(df, model, csv_path, num_patches=64):
    
    # Get image dictionary
    images = dict(df.groupby('wsi')['path'].apply(list))
    images_label = dict(df.groupby('wsi')['label'].apply(max))

    images_feature = {}
    for k, v in images.items():
        images_feature[k] = os.path.join(csv_path, k+'.csv')

    csv_data = WSICsvLoader(images_feature, images_label)
    dl = torch.utils.data.DataLoader(csv_data, batch_size=1, shuffle=False)        
    
    list_of_df = []
    with torch.no_grad():
        for i, (inputs, labels, path, pat_name) in enumerate(dl):
            output_attn, _ = model(inputs.to(device))
            output_attn = output_attn.flatten()
            df_attn = pd.DataFrame({'attention_value': output_attn.cpu().numpy(), \
                          'path': [x[0] for x in path],\
                          'label': labels.item(),\
                           'wsi': pat_name[0]})
            df_attn = df_attn.sort_values(by='attention_value', ascending=False).reset_index(drop=True).iloc[:num_patches]
            list_of_df.append(df_attn)    
            
    return pd.concat(list_of_df).reset_index(drop=True)    