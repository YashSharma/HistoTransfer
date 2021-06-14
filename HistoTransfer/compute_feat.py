import os
import torch
import pandas as pd
from tqdm import tqdm

from HistoTransfer.dataloader import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_save_feat(model, tdl, im, output_path):
    
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
        
    features_rep = torch.tensor([])
    path_list = []
    with torch.no_grad():
        for i, bs in enumerate(tdl):
            features = model(bs[0].to(device))
            features = features.view(features.shape[0], -1)
            if len(features_rep) == 0:
                features_rep = features
            else:
                features_rep = torch.cat((features_rep, features))
            path_list += list(bs[1])

    features_rep = features_rep.cpu().numpy()    
    output_path = output_path + im + '.csv'
    feat_df = pd.DataFrame(features_rep)
    feat_df['path'] = path_list
    feat_df.to_csv(output_path, index=False)
    return


def compute_feat_wsi(images, model, data_transforms, output_path, batch_size=32, net='noteffnet'):
    """ Pass all the patches in a WSI for validation prediction
    """
    
    if net != 'effnet':
        model.eval()    
    with torch.no_grad():
        for im, im_list in tqdm(images.items()):            
            td = WSIPatchLoader(im_list, transform=data_transforms)
            tdl = torch.utils.data.DataLoader(td, batch_size=batch_size,
                                             shuffle=False)
            compute_save_feat(model, tdl, im, output_path)
    return