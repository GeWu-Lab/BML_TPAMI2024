import torch
from torch import nn

class SumFusion(nn.Module):

    def __init__(self,in_c_x,out_c_x,in_c_y,out_c_y) -> None:
        super().__init__()

        self.fx=nn.Linear(in_c_x,out_c_x)
        self.fy=nn.Linear(in_c_y,out_c_y)

    def forward(self,x,y):
        out=self.fx(x)+self.fy(y)
        return x,y,out

class ConcatFusion(nn.Module):

    def __init__(self,in_c_x,in_c_y,out_c) -> None:
        super().__init__()
        self.fxy=nn.Linear(in_c_x+in_c_y,out_c)

    def forward(self,x,y):
        out=torch.cat([x,y],dim=1)
        out=self.fxy(out)
        return x,y,out

class GatedFusion(nn.Module):

    def __init__(self,in_c_x,in_c_y,mid_c,out_c,x_gate=True) -> None:
        super().__init__()

        self.fx=nn.Linear(in_c_x,mid_c)
        self.fy=nn.Linear(in_c_y,mid_c)
        self.f_out=nn.Linear(mid_c,out_c)

        self.x_gate=x_gate
        self.sigmoid=nn.Sigmoid()

    def forward(self,x,y):
        out_x=self.fx(x)
        out_y=self.fy(y)

        if self.x_gate:
            gate=self.sigmoid(out_x)
            out=self.f_out(torch.mul(gate,out_y))
        else:
            gate=self.sigmoid(out_y)
            out=self.f_out(torch.mul(out_x,gate))

        return out_x,out_y,out

from torch.autograd import Variable
from torch.nn.parameter import Parameter
class LMF(nn.Module):

    def __init__(self,rank=4,hidden_dim=512,out_dim=31,device='cuda:0'):
        super().__init__()
        self.device=device
        self.rank=rank
        self.hidden_dim=hidden_dim
        self.out_dim=out_dim
        self.x_factor=Parameter(torch.Tensor(self.rank,self.hidden_dim+1,self.out_dim)).to(device) # r,d+1,cls
        self.y_factor=Parameter(torch.Tensor(self.rank,self.hidden_dim+1,self.out_dim)).to(device)
        self.fusion_weights=Parameter(torch.Tensor(1,self.rank)).to(device)  # 1,r
        self.fusion_bias=Parameter(torch.Tensor(1,self.out_dim)).to(device)

        torch.nn.init.xavier_normal_(self.x_factor)
        torch.nn.init.xavier_normal_(self.y_factor)
        torch.nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self,x,y):
        b=x.shape[0]
        _x=torch.cat((Variable(torch.ones(b,1).to(self.device),requires_grad=False),x),dim=1) # b,d+1
        _y=torch.cat((Variable(torch.ones(b,1).to(self.device),requires_grad=False),y),dim=1)

        fusion_x=torch.matmul(_x,self.x_factor) # r,b,cls
        fusion_y=torch.matmul(_y,self.y_factor)
        fusion_zy=fusion_x*fusion_y

        output=torch.matmul(self.fusion_weights,fusion_zy.permute(1,0,2)).squeeze()+self.fusion_bias # b,cls
        # output=output.view(-1,self.out_dim)

        return output,x,y

if __name__ == "__main__":
    net=GatedFusion(10,10,10,20)
    x=torch.zeros([1,10])
    y=torch.zeros([1,10])
    x_out,y_out,z=net(x,y)
    print(x_out.shape,y_out.shape)  # torch.Size([1, 10]) torch.Size([1, 10])
    print(z.shape)  # torch.Size([1, 20])

    print(net.weight)