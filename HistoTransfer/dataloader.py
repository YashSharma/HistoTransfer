import cv2
import torch
import random
import numpy as np
import pandas as pd
import albumentations
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2, ToTensor

# DataLoader
class WSICsvLoader(Dataset):
    """
    Load WSI patch representation from csv
    """
    def __init__(self, image_path, label_dic):
        self.input_images = image_path
        self.label = label_dic
        self.id_map = dict(zip(range(len(image_path)), image_path.keys()))
    
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):
        df = pd.read_csv(self.input_images[self.id_map[idx]])
        num_feat = max([int(x) for x in df.columns if x.isdigit()])+1
        img_feat = torch.tensor(df[[str(x) for x in range(num_feat)]].values.astype(np.float32))
        label = int(self.label[self.id_map[idx]])
        img_path = df['path'].tolist()
        
        return img_feat, label, img_path, self.id_map[idx]
    
    
# Patch Loader    
class WSIPatchLoader(Dataset):
    """
    Dataloader for iterating through all patches in a WSI
    """    
    def __init__(self, image_path, transform=None):
        self.input_images = image_path
        self.transform = transform  

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        im_path = self.input_images[idx]
        im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
        if self.transform:
            im = self.transform(image=im)['image']            
        return im, self.input_images[idx]
    
# Return Selected Patches x 3 x 512 x 512
class WSIDataloader(Dataset):
    """
    Dataloader for sampling instance from each cluster in a WSI 
    """
    def __init__(self, image_path, label_dic, transform=None, sample_batch=False):
        self.input_images = image_path
        self.label = label_dic
        self.id_map = dict(zip(range(len(image_path)), image_path.keys()))
        self.transform = transform  
        self.sample_batch = sample_batch

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image_list = []
        for im_id, im_path in enumerate(self.input_images[self.id_map[idx]]):
            im = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
            image_list.append(im)
                
        if self.transform:
            for im_id, im in enumerate(image_list):
                # Albumentation added
                image_list[im_id] = self.transform(image=im)['image']
                
        if self.sample_batch:
            random.shuffle(image_list)
            image_list = image_list[:8]
            
        image = torch.stack(image_list)
        label = int(self.label[self.id_map[idx]])

        return image, label      