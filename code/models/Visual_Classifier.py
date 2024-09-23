from torch import nn
import torch.nn.functional as F
import torch

from .Resnet_18 import resnet18

class Classifier(nn.Module):

    def __init__(self,cfg,device='cuda:0'):
        super().__init__()

        self.encoder_2=resnet18(modality=cfg.modality[1])

        self.cfg=cfg
        self.device=device

        self.linear=nn.Linear(512,31)

    def forward(self,mod_1):
        B=mod_1.shape[0]
        out_1=self.encoder_2(mod_1)   
        
        _,C,H,W=out_1.shape
        out_1=out_1.reshape(B,-1,C,H,W).permute(0,2,1,3,4)
        out_1=F.adaptive_avg_pool3d(out_1,1)
        out_1=torch.flatten(out_1,1)

        out_1=self.linear(out_1)
        return out_1
        
    

if __name__ == "__main__":
    pass
