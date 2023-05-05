import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

print('all imported')
EPOCHS=51
BATCH_SIZE=4
ROOT='data'
LEARNING_RATE=2e-4
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
LOAD=False
L1_LAMBDA=100

both=A.Compose(
    [A.Resize(height=256,width=256),A.HorizontalFlip(p=0.5),],
    additional_targets={'image0':'image'}
    )
transform_input=A.Compose(
    [
        A.ColorJitter(p=0.1),
        A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],max_pixel_value=255.0),
        ToTensorV2(),
        ]
    )

transform_mask=A.Compose(
    [
        
        A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],max_pixel_value=255.0),
        ToTensorV2(),
        ]
    )

if __name__=='__main__':
    def pil(image):
        trans=transforms.Compose([
            transforms.ToPILImage(),
            ])
        return trans(image)
    img=np.array(Image.open('data/test/1000.jpg'))
    x,y=img[:,:600],img[:,600:]
    print(type(x),type(y))
    aug=both(image=x,image0=y)
    x,y=aug['image'],aug['image0']
    pil(x).show()
    pil(y).show()
    x=transform_input(image=x)['image']
    y=transform_mask(image=y)['image']
    pil(x).show()
    pil(y).show()













    


