from torch import nn
import torch.nn.functional as F

from .Resnet_18 import resnet18

class Classifier(nn.Module):

    def __init__(self,cfg,device='cuda:0'):
        super().__init__()

        self.encoder_1=resnet18(modality=cfg.modality[0])

        self.cfg=cfg
        self.device=device

        self.linear=nn.Linear(512,31)

    def forward(self,mod_1):
        out_1=self.encoder_1(mod_1)   
        out_1=F.adaptive_avg_pool2d(out_1,1)
        out_1=out_1.squeeze(2).squeeze(2)  # [B,2048]

        out_1=self.linear(out_1)
        return out_1
        
    

if __name__ == "__main__":
    pass
