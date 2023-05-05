from PIL import Image
import numpy as np
import os
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import config


class MapDataset(Dataset):
    def __init__(self,root,train):
        self.root=f'{root}/train' if train else f'{root}/test'
        self.list_dir=os.listdir(self.root)
    def __getitem__(self,index):
        img_path=os.path.join(self.root,self.list_dir[index])
        img=np.array(Image.open(img_path))
        x,y=img[:,:600],img[:,600:]
        
        aug=config.both(image=x,image0=y)
        x,y=aug['image'],aug['image0']
        x=config.transform_input(image=x)['image']
        y=config.transform_mask(image=y)['image']
        
        return x,y
        
    def __len__(self):
        return len(self.list_dir)
    
def trans(image):
    pass
def run():
    root='data'
    dataset=MapDataset(root,train=False)
    dataloader=DataLoader(dataset=dataset,batch_size=1,shuffle=True)
    for i,j in enumerate(dataloader):
        x,y=j[0],j[1]
        print(type(x),type(y),x.shape)
        x,y=x.reshape(3,256,256),y.reshape(3,256,256)
        trans=transforms.ToPILImage()
        x,y=trans(x),trans(y)
        x.show()
        y.show()
        
        break
if __name__=='__main__':
    run()
