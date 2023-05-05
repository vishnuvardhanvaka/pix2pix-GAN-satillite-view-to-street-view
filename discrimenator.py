
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
import sys

class conv_block(nn.Module):
    def __init__(self,inc,outc,stride=2):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(inc,outc,kernel_size=4,stride=stride,padding=1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(outc),
            nn.LeakyReLU(0.2),
            )
    def forward(self,x):
        return self.conv1(x)

#DISCRIMENATOR...
class Discrimenator(nn.Module):
    def __init__(self,inc=3,features=[64,128,256,512]):
        super().__init__()
        self.initial=nn.Sequential(
            nn.Conv2d(inc*2,features[0],kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            )
        layers=[]
        inc=features[0]
        for feature in features[1:]:
            if feature==512:
                stride=1
            else:
                stride=2
            layers.append(conv_block(inc,feature,stride))
            inc=feature
        layers.append(
            nn.Conv2d(in_channels=inc,out_channels=1,kernel_size=4,stride=1,padding=1,padding_mode='reflect')
            )
        self.conv=nn.Sequential(*layers)
        
                        
    def forward(self,x,y):
        x=torch.cat([x,y],dim=1)
        x=self.initial(x)
        x=self.conv(x)
        return x

if __name__=='__main__':
    size=256
    def transform(image):
        trans=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size,size)),
            ])
        return trans(image)
    def pil(image):
        trans=transforms.ToPILImage()
        return trans(image)
    img=Image.open('data/car1.jpg')
    img=transform(img)
    
    x=torch.randn(1,3,size,size)
    y=torch.randn(1,3,size,size)
    obj=Discrimenator(inc=3)
    r=obj(x,y)
    print(r.shape,r)
    #img=pil(r)
    #img.show()
    
        


























