
import argparse
import torch
import numpy as np
import random
import json
import os
from os.path import join
import sys
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.Classifier import Classifier
from config import Config
from dataset.KS import KSDataset
from utils.log_file import Logger
from datetime import datetime

from dataset.spatial_transforms import get_spatial_transform,get_val_spatial_transforms
from dataset.loader import VideoLoaderHDF5

from sklearn.metrics import average_precision_score
import torch.nn.functional as F


TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now()) 


def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--use_modulation',action='store_true',help='use gradient modulation')
    parser.add_argument('--use_adam_drop',action='store_true',help='use adam-drop')
    parser.add_argument('--modulation', default='OGM_GE', type=str,choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--use_OGM_plus',action='store_true')
    parser.add_argument('--fusion_method', default='concat', type=str,choices=['sum', 'concat', 'gated'])
    parser.add_argument('--train', action='store_true', help='turn on train mode')
    parser.add_argument('--resume_model',action='store_true',help='whether to resume model')
    parser.add_argument('--resume_model_path')
    parser.add_argument('--q_base',type=float,default=0.5)
    parser.add_argument('--lam',type=float,default=0.5)
    parser.add_argument('--p_exe',type=float,default=0.7)
    parser.add_argument('--alpha',type=float,default=1.0)
    parser.add_argument('--modulation_starts',type=int,default=0)
    parser.add_argument('--modulation_ends',type=int,default=80)
    parser.add_argument('--audio_drop',type=float,default=0.0)
    parser.add_argument('--visual_drop',type=float,default=0.0)
    parser.add_argument('--exp_name',type=str,default='exp')

    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

weight_a=0.36
weight_v=0.27
weight_av=0.37

def train(cfg,epoch,model,device,dataloader,optimizer,scheduler,tb=None):
    loss_fn=nn.CrossEntropyLoss().to(device)
    relu=nn.ReLU(inplace=True)
    tanh=nn.Tanh()
    model.train()
    total_loss=0
    total_loss_1=0
    total_loss_2=0
    with tqdm(total=len(dataloader), desc=f"Train-epoch-{epoch}") as pbar:
        for step, (spec,image,label) in  enumerate(dataloader):
            spec=spec.to(device) # b,h,w
            image=image.to(device) # b,c,t,h,w
            label=label.to(device)
            optimizer.zero_grad()
            warm_up=1 if epoch<=5 else 0
            # warm_up=0
            out_1,out_2,out,update_flag,performance_1,performance_2=model(spec.unsqueeze(1).float(),image.float(),label,warm_up)

            if warm_up==0 and cfg.use_adam_drop:
                if torch.sum(update_flag,dim=0)==0:
                    continue
                select_mask=update_flag!=0
                label=label[select_mask]
                out_1=out_1[select_mask]
                out_2=out_2[select_mask]
            

            loss=loss_fn(out,label)
            loss_1=loss_fn(out_1,label)
            loss_2=loss_fn(out_2,label)
            total_loss+=loss.item()
            total_loss_1+=loss_1.item()
            total_loss_2+=loss_2.item()

            # if warm_up==0:
            #     loss=loss*weight_av+loss_1*weight_a+loss_2*weight_v

            loss.backward()

            if warm_up==0 and cfg.use_modulation:
                # log.logger.info('per_1:{}  per_2:{} '.format(performance_1,performance_2))
                coeff_1,coeff_2=None,None
                radio_1=performance_1/performance_2
                radio_2=performance_2/performance_1
                # if cfg.form=='/':
                #     radio_1=performance_1/performance_2
                #     radio_2=performance_2/performance_1
                # else:
                #     radio_1=performance_1-performance_2
                #     radio_2=performance_2-performance_1

                if cfg.use_OGM_plus:
                    if radio_1>1:
                        # coeff_2=1+tanh(cfg.alpha*relu(radio_1))
                        coeff_2=4
                        coeff_1=1
                    else:
                        coeff_2=1
                        # coeff_1=1+tanh(cfg.alpha*relu(radio_2))
                        coeff_1=4
                else:
                    if radio_1>1:
                        coeff_1=1-tanh(cfg.alpha*relu(radio_1))
                        # if cfg.func=='tanh':
                        #     coeff_1=1-tanh(cfg.alpha*relu(radio_1))
                        # else:
                        #     coeff_1=1-sigmoid(cfg.alpha*relu(radio_1))

                        coeff_2=1
                    else:
                        coeff_1=1
                        coeff_2=1-tanh(cfg.alpha*relu(radio_2))
                        # if cfg.func=='tanh':
                        #     coeff_2=1-tanh(cfg.alpha*relu(radio_2))
                        # else:
                        #     coeff_2=1-sigmoid(cfg.alpha*relu(radio_2))

                if cfg.modulation_starts<=epoch<=cfg.modulation_ends:
                    for name,parms in model.named_parameters():
                        layer_name=str(name).split('.')[0]
                        if 'encoder_1' in layer_name and parms.grad is not None and len(parms.grad.size()) == 4:
                            if cfg.modulation == 'OGM_GE':  
                                parms.grad = parms.grad * coeff_1 + \
                                            torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                            elif cfg.modulation == 'OGM':
                                parms.grad *= coeff_1

                        if 'encoder_2' in layer_name and parms.grad is not None and len(parms.grad.size()) == 4:
                            if cfg.modulation == 'OGM_GE':  
                                parms.grad = parms.grad * coeff_2 + \
                                            torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                            elif cfg.modulation == 'OGM':
                                parms.grad *= coeff_2

            optimizer.step() 
            pbar.update(1)
        
        scheduler.step()

    return total_loss/len(dataloader),total_loss_1/len(dataloader),total_loss_2/len(dataloader)


def val(model,device,dataloader):
    softmax=nn.Softmax(dim=1)
    sum_all=0
    sum_1=0
    sum_2=0
    tot=0
    all_out=[]
    all_label=[]
    with torch.no_grad():
        model.eval()
        for step,(spec,img,label) in enumerate(dataloader):
            spec=spec.to(device)
            img=img.to(device)
            label=label.to(device)
            out_1,out_2,out,update_flag,performance_1,performance_2=model(spec.unsqueeze(1).float(),img.float(),label,warm_up=1)
            prediction=softmax(out)
            pred_1=softmax(out_1)
            pred_2=softmax(out_2)
            tot+=img.shape[0]
            sum_all+=torch.sum(torch.argmax(prediction,dim=1)==label).item()
            sum_1+=torch.sum(torch.argmax(pred_1,dim=1)==label).item()
            sum_2+=torch.sum(torch.argmax(pred_2,dim=1)==label).item()
            
            for i in range(label.shape[0]):
                all_out.append(prediction[i].cpu().data.numpy())
                ss=torch.zeros(31)
                ss[label[i]]=1
                all_label.append(ss.numpy())
        
        all_out=np.array(all_out)
        all_label=np.array(all_label)
        mAP=average_precision_score(all_label,all_out)
        

    return mAP,sum_all/tot,sum_1/tot,sum_2/tot


def write2txt(fp,info,mode='a'):
    with open(fp,mode=mode) as f:
        f.write(info)
        f.write('\n')


def main():
    # job_id=datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    cfg = Config()
    args=get_arguments()
    cfg.parse(vars(args))
    setup_seed(cfg.random_seed)

    job_name=args.exp_name
    cur_dir=os.path.join('results',job_name)
    os.makedirs(cur_dir,exist_ok=True)
    
    # log=Logger(os.path.join(cur_dir,'log.log'),level='info')
    writer=None
    if cfg.use_tensorboard:
        writer_path=os.path.join(cur_dir,'tensorboard')
        os.makedirs(writer_path,exist_ok=True)
        writer=SummaryWriter(writer_path)

    saved_data=vars(cfg)
    cmd=' '.join(sys.argv)
    saved_data.update({'cmd':cmd})
    saved_data=json.dumps(saved_data,indent=4)
    with open(os.path.join(cur_dir,'config.json'),'w') as f:
        f.write(saved_data)
    
    device=torch.device('cuda')

    spatial_transforms=get_spatial_transform(opt=cfg)
    val_spatial_transforms=get_val_spatial_transforms(opt=cfg)
    train_dataset=KSDataset(mode='training',spatial_transform=spatial_transforms,video_loader=VideoLoaderHDF5())
    test_dataset=KSDataset(mode='testing',spatial_transform=val_spatial_transforms,video_loader=VideoLoaderHDF5(),audio_drop=cfg.audio_drop,visual_drop=cfg.visual_drop)

    train_loader=DataLoader(train_dataset,batch_size=cfg.batch_size,shuffle=True,num_workers=32,pin_memory=True)
    test_loader=DataLoader(test_dataset,batch_size=cfg.batch_size,shuffle=False,num_workers=32,pin_memory=True)

    model=Classifier(cfg,device=device)

    if cfg.resume_model:
        state_dict=torch.load(cfg.resume_model_path,map_location='cuda')
        model.load_state_dict(state_dict=state_dict)
    else:
        model.apply(weight_init)
    
    model.to(device)

    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=0.01)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=cfg.lr_decay_step,gamma=cfg.lr_decay_ratio)

    start_epoch=-1
    best_acc=0.0
    logger_path=join(cur_dir,'log.txt')

    if cfg.train:
        for epoch in range(start_epoch+1,cfg.epochs):
            loss,loss_1,loss_2=train(cfg,epoch,model,device,train_loader,optimizer,scheduler,tb=writer)
            mAP,acc,acc_1,acc_2=val(model,device,test_loader)
            # log.logger.info('epoch:{} acc:{:.4f} acc_1:{:.4f} acc_2:{:.4f} mAP:{:.4f}'.format(epoch,acc,acc_1,acc_2,mAP))
            write2txt(fp=logger_path,info=f'epoch:{epoch} acc:{acc:.4f} acc_1:{acc_1:.4f} acc_2:{acc_2:.4f} mAP:{mAP:.4f}')
            if writer is not None:
                writer.add_scalars(main_tag='Loss',tag_scalar_dict={'loss':loss,'loss_1':loss_1,'loss_2':loss_2},global_step=epoch)
                writer.add_scalars(main_tag='Acc',tag_scalar_dict={'acc':acc,'acc_1':acc_1,'acc_2':acc_2},global_step=epoch)

            if acc>best_acc:
                best_acc=acc
                saved_data={}
                saved_data['epoch']=epoch
                saved_data['acc']=acc
                saved_data['mAP']=mAP
                saved_data['acc_1']=acc_1
                saved_data['acc_2']=acc_2
                saved_data=json.dumps(saved_data,indent=4)

                with open(os.path.join(cur_dir,'best_model.json'),'w') as f:
                    f.write(saved_data)

                torch.save(model.state_dict(),os.path.join(cur_dir,'best_model.pth'))
    else:
        mAP,acc,acc_1,acc_2=val(model,device,test_loader)
        print('mAP:{} Acc:{}'.format(mAP,acc))
    

if __name__ == "__main__":
    main()
