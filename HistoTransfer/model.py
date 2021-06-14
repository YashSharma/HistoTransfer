import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# Model    
class WSIFeatClassifier(nn.Module):
    def __init__(self, n_class=2, feat_dim=512):
        super(WSIFeatClassifier, self).__init__()
        self.L = 64
        self.D = 32
        self.K = 1
        self.tail = nn.Sequential(nn.Linear(feat_dim, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, self.L),
                                  nn.ReLU())
        self.attention = nn.Sequential(nn.Linear(self.L, self.D),
                                        nn.Tanh(),
                                        nn.Linear(self.D, self.K))
        
        self.classifier = nn.Sequential(nn.Linear(self.L*self.K, n_class))
        self.patch_classifier = nn.Sequential(nn.Linear(self.L*self.K, n_class))

    def forward(self, x):
        x = x.squeeze(0)
        x = self.tail(x)
        xp = self.patch_classifier(x)
        
        A_unnorm = self.attention(x)
        A = torch.transpose(A_unnorm, 1, 0)
        A = F.softmax(A, dim=1)
        
        M = torch.mm(A, x)
        Y_prob = self.classifier(M)        
        
        return Y_prob, xp, A_unnorm 
        
# Get Embedding Representation, Attn Value
class EncAttn(nn.Module):
    def __init__(self, model_base, head=False):
        super(EncAttn, self).__init__()
        self.head = head
        if head:
            self.head = model_base.head
        self.tail = model_base.tail
        self.attention = model_base.attention
        
    def forward(self, x):
        if self.head:
            x = self.head(x)
            x = x.view(x.size(0), -1)
        x = self.tail(x)
        attn = self.attention(x)
        
        return attn, x        
    
class WSIClassifier(nn.Module):
    def __init__(self, n_class=2, feat_dim=512, base_model='resnet18', backbone=None):
        super(WSIClassifier, self).__init__()
        self.L = 64
        self.D = 32
        self.K = 1
        
        if base_model == 'resnet18':
            backbone = models.resnet18(pretrained=True)
        elif base_model == 'resnet34':
            backbone = models.resnet34(pretrained=True)
        elif base_model == 'resnet50':
            backbone = models.resnet50(pretrained=True)
        elif base_model == 'densenet121':
            backbone = models.densenet121(pretrained=True)
        elif base_model == 'self':
            backbone = backbone

        modules = list(backbone.children())[:-1]          
        self.head = nn.Sequential(*modules)
        self.tail = nn.Sequential(nn.Linear(feat_dim, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, self.L),
                                        nn.ReLU())

        self.attention = nn.Sequential(nn.Linear(self.L, self.D),
                                        nn.Tanh(),
                                        nn.Linear(self.D, self.K))
        
        self.classifier = nn.Sequential(nn.Linear(self.L*self.K, n_class))
        self.patch_classifier = nn.Sequential(nn.Linear(self.L*self.K, n_class))
        
    def forward(self, x):
        x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1])
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.tail(x)        
        xp = self.patch_classifier(x)
        
        A_unnorm = self.attention(x)
        A = torch.transpose(A_unnorm, 1, 0)
        A = F.softmax(A, dim=1)
        
        M = torch.mm(A, x)
        Y_prob = self.classifier(M)        
        return Y_prob, xp, A_unnorm          