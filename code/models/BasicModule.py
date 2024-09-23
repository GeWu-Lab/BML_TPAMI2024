
import torch
from torch import nn

import time

class BasicModule(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.model_name=str(type(self))

    def load(self,path):
        self.load_state_dict(torch.load(path))

    def save(self,name=None):
        if name is None:
            name=time.strftime('checkpoints/'+self.model_name+'_'+'%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(),name)

        return name

if __name__ == "__main__":
    pass
