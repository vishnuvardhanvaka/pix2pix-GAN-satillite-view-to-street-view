import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class conv_block(nn.Module):
    def __init__(self,inc,outc,act='relu',down=True,use_dropout=False):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(inc,outc,kernel_size=4,stride=2,padding=1,padding_mode='reflect',bias=False) if down else nn.ConvTranspose2d(inc,outc,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU() if act=='relu' else nn.LeakyReLU(0.2),
            )
        self.dp=use_dropout
        self.dropout=nn.Dropout(0.5)
    def forward(self,x):
        x=self.conv1(x)
        x=self.dropout(x) if self.dp else x
        return x
class Generator(nn.Module):
    def __init__(self,inc=3):
        super().__init__()

        self.initial=nn.Sequential(
            nn.Conv2d(inc,64,kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            )
        self.down1=conv_block(64,128,act='leaky')
        self.down2=conv_block(128,256,act='leaky')
        self.down3=conv_block(256,512,act='leaky')
        self.down4=conv_block(512,512,act='leaky')
        self.down5=conv_block(512,512,act='leaky')
        self.down6=conv_block(512,512,act='leaky')

        self.bn=nn.Sequential(
            nn.Conv2d(512,512,kernel_size=4,stride=2,padding=1,padding_mode='reflect'),
            nn.ReLU(),
            )

        self.up1=conv_block(512,512,act='relu',down=False,use_dropout=True)
        self.up2=conv_block(1024,512,act='relu',down=False,use_dropout=True)
        self.up3=conv_block(1024,512,act='relu',down=False,use_dropout=True)
        self.up4=conv_block(1024,512,act='relu',down=False,use_dropout=False)
        self.up5=conv_block(1024,256,act='relu',down=False,use_dropout=False)
        self.up6=conv_block(512,128,act='relu',down=False,use_dropout=False)
        self.up7=conv_block(256,64,act='relu',down=False,use_dropout=False)

        self.final=nn.Sequential(
            nn.ConvTranspose2d(128,inc,4,2,1),
            nn.Tanh(),
            )
    def forward(self,x):

        #DECODING...
        d1=self.initial(x)
        d2=self.down1(d1)
        d3=self.down2(d2)
        d4=self.down3(d3)
        d5=self.down4(d4)
        d6=self.down5(d5)
        d7=self.down6(d6)
        
        bn=self.bn(d7)

        up1=self.up1(bn)
        up2=self.up2(torch.cat([up1,d7],dim=1))
        up3=self.up3(torch.cat([up2,d6],1))
        up4=self.up4(torch.cat([up3,d5],1))
        up5=self.up5(torch.cat([up4,d4],1))
        up6=self.up6(torch.cat([up5,d3],1))
        up7=self.up7(torch.cat([up6,d2],1))

        final=self.final(torch.cat([up7,d1],1))
        
        return final
def transform(image):
    trans=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,256)),
        ])
    return trans(image)
def show(image):
    trans=transforms.Compose([
        transforms.ToPILImage(),
        ])
    image=trans(image)
    image.show()
def test():
    a=torch.randn(1,1,256,256)
    img=np.array(Image.open('data/test/1023.jpg'))
    x,y=img[:,:600],img[:,600:]
    print(x.shape,y.shape)
    x,y=transform(x),transform(y)
    x,y=x.reshape(1,3,256,256),y.reshape(1,3,256,256)
    gen=Generator(3)
    
    gen_load=torch.load('gen.pth')
    
    gen.load_state_dict(gen_load['gen'])
    
    r=gen(y)
    print(r.shape)
    r=r.reshape(3,256,256)
    show(r)

if __name__=='__main__':
    test()
        

















