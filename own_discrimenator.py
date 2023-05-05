import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self,inc,outc,stride=2):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(inc,outc,kernel_size=4,stride=stride,padding=1,padding_mode='reflect'),
            nn.BatchNorm2d(outc),
            nn.LeakyReLU(0.2),
            )
    def forward(self,x):
        x=self.conv1(x)
        return x

class Discrimenator(nn.Module):
    def __init__(self,inc=3):
        super().__init__()
        self.initial=nn.Sequential(
            nn.Conv2d(inc*2,64,4,2,1,padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            )
        self.conv=nn.Sequential(
            conv_block(64,128),
            conv_block(128,256),
            conv_block(256,512,stride=1),
            conv_block(512,1,stride=1),
            
            )
    def forward(self,x,y):
        cat=torch.cat([x,y],dim=1)
        x=self.initial(cat)
        x=self.conv(x)
        return x

def test():
    x=torch.randn(1,3,256,256)
    y=torch.randn(1,3,256,256)
    obj=Discrimenator(3)
    r=obj(x,y)
    print(r,r.shape)
if __name__=='__main__':
    test()
    
    
        
        







