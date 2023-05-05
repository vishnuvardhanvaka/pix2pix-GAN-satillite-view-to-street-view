import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import sys
from PIL import Image
from torch.utils.data import DataLoader
from dataset import MapDataset
from generator import Generator
from discrimenator import Discrimenator
import config
from tqdm import tqdm

#TRANSFORMS....
def transform(image):
    trans=transforms.Compose([
        transforms.ToPILImage(),
        ])
    return trans(image)
def show(image):
    image=transform(image)
    image.show()
#TRAINING FUNCTION...
def train(train_loader,dis,gen,dis_opt,gen_opt,bce,l1_loss):
    #loop=tqdm(train_loader,leave=True)
    for i,(x,y) in enumerate(train_loader):
        #print(i,x.shape,y.shape)
        x,y=x.to(config.DEVICE),y.to(config.DEVICE)
        fake=gen(x)

        #TRAINING DISCRIMENATOR...
        dis_fake=dis(x,fake)
        dis_real=dis(x,y)
        dis_fake_loss=bce(dis_fake,torch.zeros_like(dis_fake))
        dis_real_loss=bce(dis_real,torch.ones_like(dis_real))
        dis_loss=(dis_fake_loss+dis_real_loss)/2
        dis_opt.zero_grad()
        dis_loss.backward(retain_graph=True)
        dis_opt.step()

        #TRAINING GENERATOR...
        dis_fake=dis(x,fake)
        gen_fake_loss=bce(dis_fake,torch.ones_like(dis_fake))
        gen_l1_loss=l1_loss(fake,y)*config.L1_LAMBDA
        gen_loss=gen_fake_loss+gen_l1_loss

        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()
        
        return dis_loss,gen_loss
        

#LOADING DATA....

train_dataset=MapDataset(config.ROOT,train=True)
test_dataset=MapDataset(config.ROOT,train=False)

train_loader=DataLoader(dataset=train_dataset,batch_size=config.BATCH_SIZE,shuffle=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=1,shuffle=False)

#MODEL
dis=Discrimenator(inc=3).to(config.DEVICE)
gen=Generator(inc=3).to(config.DEVICE)

dis_opt=torch.optim.Adam(dis.parameters(),lr=config.LEARNING_RATE,betas=(0.5,0.9999))
gen_opt=torch.optim.Adam(gen.parameters(),lr=config.LEARNING_RATE,betas=(0.5,0.9999))

bce=nn.BCEWithLogitsLoss()
l1_loss=nn.L1Loss()

#g_scaler=torch.cuda.amp.GradScaler()
#d_scaler=torch.cuda.amp.GradScaler()
if config.LOAD:
    dis_load=torch.load('dis1.pth')
    gen_load=torch.load('gen1.pth')
    dis.load_state_dict(dis_load['dis'])
    gen.load_state_dict(gen_load['gen'])
    dis_opt.load_state_dict(dis_load['opt'])
    gen_opt.load_state_dict(gen_load['opt'])
    

for epoch in range(config.EPOCHS):
    loss=train(train_loader,dis,gen,dis_opt,gen_opt,bce,l1_loss)
    if epoch%1==0:
        print(f' EPOCH ={epoch} LOSS= dis({loss[1]} gen({loss[0]}))')
    if (epoch+1)%1==0:
        for i,(x,y) in enumerate(test_loader):
            
            t=gen(x)*0.5+0.5
            break
        show(t[0])

    
    
dd={
    'dis':dis.state_dict(),
    'opt':dis_opt.state_dict()
    }
dg={
    'gen':gen.state_dict(),
    'opt':gen_opt.state_dict()
    }
torch.save(dd,'dis1.pth')
torch.save(dg,'gen1.pth')
print('saved....')
    






















