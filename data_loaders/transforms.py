import torch
from torchvision import transforms as T

def image_transform(size,train=False):
    operations = ([T.RandomResizedCrop(size,scale=(.8,1.))] if train else
                  [T.Resize((size,size))])
    return T.Compose(operations+[T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])

class StochasticPixelMask:
    def __init__(self,s=.001):
        if not 0<=s<1:
            raise ValueError('mask probability must be in [0,1)')
        self.s = s

    def __call__(self,x):
        # One pixel mask per image, shared across RGB channels (KT Eq.10).
        shape = (*x.shape[:-3],1,*x.shape[-2:]) if x.ndim>=3 else x.shape
        return x*(torch.rand(shape,device=x.device)>=self.s).to(x.dtype)
