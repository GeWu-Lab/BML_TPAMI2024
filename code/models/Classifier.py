import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from .fusion_model import ConcatFusion,SumFusion,GatedFusion,LMF
from .Resnet_18 import resnet18

class custom_autograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx,input,theta):
        ctx.save_for_backward(input,theta)
        return input/(1-theta.item())

    @staticmethod
    def backward(ctx,grad_output):
        input,theta=ctx.saved_tensors
        input_grad=1/(1-theta.item())*grad_output.clone()

        return input_grad,theta


class Modality_drop():

    def __init__(self,dim_list,p_exe=0.7,device='cuda'):
        self.dim_list=dim_list
        self.p_exe=p_exe
        self.device=device

    def execute_drop(self,fead_list,q):
        B = fead_list[0].shape[0]
        D = fead_list[0].shape[1]
        exe_drop = torch.tensor(np.random.rand(1)).to(device=self.device) >= 1-self.p_exe
        if not exe_drop:
            return fead_list, torch.ones([B],dtype=torch.int32,device=self.device)

        num_mod=len(fead_list)
        d_sum=sum(self.dim_list)
        q_sum=sum(self.dim_list*q)
        theta=q_sum/d_sum
        # p_sum=sum(self.dim_list*(1-q))
        # theta=p_sum/d_sum

        mask=torch.distributions.Bernoulli(1-q).sample([B,1]).permute(2,1,0).contiguous().reshape(num_mod,B,-1).to(device=self.device) # [2,B,1]
        # print(f'mask:{mask}')
        concat_list=torch.stack(fead_list,dim=0)   # [2,B,D]
        concat_list=torch.mul(concat_list,mask)
        concat_list=custom_autograd.apply(concat_list,theta)
        mask=torch.transpose(mask,0,1).squeeze(-1)  # [B,2]
        update_flag=torch.sum(mask,dim=1)>0  
        cleaned_fea=torch.masked_select(concat_list,update_flag.unsqueeze(-1)).reshape(num_mod,-1,D)
        cleaned_fea=torch.chunk(cleaned_fea,num_mod,dim=0) ]
        cleaned_fea=[_.squeeze(0) for _ in cleaned_fea]   # [B,D]
        return cleaned_fea,update_flag


def calcu_q(performance_1,performance_2,q_base,fix_lambda):
    q=torch.tensor([0.0,0.0])
    relu = nn.ReLU(inplace=True)
    ratio_1=torch.tanh(relu(performance_1/performance_2-1))
    ratio_2=torch.tanh(relu(performance_2/performance_1-1))
    
    lamda = fix_lambda

    
    q[0] = q_base * (1 + lamda * ratio_1) if ratio_1>0 else 0
    q[1] = q_base * (1 + lamda * ratio_2) if ratio_2>0 else 0
    
    q=torch.clip(q,0.0,1.0)
    
    return q


class Classifier(nn.Module):

    def __init__(self,cfg,device='cuda'):
        super().__init__()

        self.encoder_1=resnet18(modality='audio')
        self.encoder_2=resnet18(modality='visual')

        self.cfg=cfg
        self.device=device

        self.softmax=nn.Softmax(dim=1)
        self.fusion_model=ConcatFusion(in_c_x=512,in_c_y=512,out_c=31)
       
        if self.cfg.use_adam_drop:
            self.modality_drop=Modality_drop(dim_list=torch.tensor(self.cfg.d),p_exe=self.cfg.p_exe,device=self.device)
            

    def forward(self,mod_1,mod_2,label,warm_up=1):
        out_1=self.encoder_1(mod_1)   
        out_2=self.encoder_2(mod_2)  # [B,T,C,H,W]--> [B,2048,2,2]

        _,C,H,W=out_2.shape
        B=out_1.shape[0]

        out_2=out_2.reshape(B,-1,C,H,W).permute(0,2,1,3,4)

        out_1=F.adaptive_avg_pool2d(out_1,1)
        out_2=F.adaptive_avg_pool3d(out_2,1)

        out_1=out_1.squeeze(2).squeeze(2)  # [B,2048]
        out_2=out_2.squeeze(2).squeeze(2).squeeze(2)          # [B,2048]

        performance_1=None
        performance_2=None
        t1,t2=None,None
        
        w=self.fusion_model.fxy.weight.clone().detach()
        b=self.fusion_model.fxy.bias.clone().detach()
        
        # if self.cfg.t1_bias==0.5:
        #     t1_bias=b/2
        # elif self.cfg.t1_bias==0.0:
        #     t1_bias=0.0
        # elif self.cfg.t1_bias==0.3:
        #     t1_bias=b/3
        # elif self.cfg.t1_bias==0.6:
        #     t1_bias=2*b/3
        # elif self.cfg.t1_bias==1.0:
        #     t1_bias=b
        t1_bias=b/2

        # if self.cfg.t2_bias==0.5:
        #     t2_bias=b/2
        # elif self.cfg.t2_bias==0.0:
        #     t2_bias=0.0
        # elif self.cfg.t2_bias==0.3:
        #     t2_bias=b/3
        # elif self.cfg.t2_bias==0.6:
        #     t2_bias=2*b/3
        # elif self.cfg.t2_bias==1.0:
        #     t2_bias=b
        t2_bias=b/2

        t1=torch.mm(out_1,torch.transpose(w[:,:512],0,1))+t1_bias
        t2=torch.mm(out_2,torch.transpose(w[:,512:],0,1))+t2_bias
        
        performance_1=sum([self.softmax(t1)[i][int(label[i].item())] for i in range(t1.shape[0])])
        performance_2=sum([self.softmax(t2)[i][int(label[i].item())] for i in range(t2.shape[0])])
      
        if warm_up==0 and self.cfg.use_adam_drop:
            self.q=calcu_q(performance_1,performance_2,self.cfg.q_base,fix_lambda=self.cfg.lam)
            cleaned_fea,update_flag=self.modality_drop.execute_drop([out_1,out_2],self.q)
            cleaned_fae_1,cleaned_fea_2,out=self.fusion_model(cleaned_fea[0],cleaned_fea[1])
            return t1,t2,out,update_flag,performance_1,performance_2

        else:
            x,y,out=self.fusion_model(out_1,out_2)
            return t1,t2,out,torch.ones([B],dtype=torch.int32,device=self.device),performance_1,performance_2



class AVClassifier_gb(nn.Module):
    def __init__(self, n_classes):
        super(AVClassifier_gb, self).__init__()
        self.n_classes = n_classes

        self.encoder_1=resnet18(modality='audio')
        self.encoder_2=resnet18(modality='visual')
        
        self.fusion_model = ConcatFusion(512,512,31)
        
        self.audio_head = nn.Linear(512, n_classes)
        self.visual_head = nn.Linear(512, n_classes)


    def forward(self, audio, visual):
        out_1=self.encoder_1(audio)   
        out_2=self.encoder_2(visual)  # [B,T,C,H,W]--> [B,2048,2,2]

        _,C,H,W=out_2.shape
        B=out_1.shape[0]

        out_2=out_2.reshape(B,-1,C,H,W).permute(0,2,1,3,4)

        out_1=F.adaptive_avg_pool2d(out_1,1)
        out_2=F.adaptive_avg_pool3d(out_2,1)

        out_1=out_1.squeeze(2).squeeze(2)  # [B,2048]
        out_2=out_2.squeeze(2).squeeze(2).squeeze(2)

        x,y,out=self.fusion_model(out_1,out_2)
        return x,y,out

